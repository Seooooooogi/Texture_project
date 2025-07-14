# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from transformers import CLIPFeatureExtractor, CLIPModel, CLIPTokenizer, CLIPProcessor, CLIPTextModel, CLIPVisionModelWithProjection, CLIPImageProcessor, CLIPTextModelWithProjection
from tqdm import tqdm
import sys
import os
sys.path.append('./adapter')
sys.path.append('./')
from PIL import Image
from threestudio.models.guidance.sd_step import *

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.ops import perpendicular_component
from threestudio.utils.typing import *

from threestudio.models.guidance.adapter.ip_adapter.ip_adapter import *

class ToWeightsDType(nn.Module):
    def __init__(self, module: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return self.module(x).to(self.dtype)
        
# +
@threestudio.register("stable-diffusion-ip-guidance")
class StableDiffusionIPGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 100.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        seed: int = 0
        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        sqrt_anneal: bool = False  # sqrt anneal proposed in HiFA: https://hifa-team.github.io/HiFA-site/
        trainer_max_steps: int = 25000
        use_img_loss: bool = False  # image-space SDS proposed in HiFA: https://hifa-team.github.io/HiFA-site/

        weighting_strategy: str = "sds"

        token_merging: bool = False
        token_merging_params: Optional[dict] = field(default_factory=dict)

        view_dependent_prompting: bool = True

        """Maximum number of batch items to evaluate guidance for (for debugging) and to save on disk. -1 means save all items."""
        max_items_eval: int = 4
        use_plus: bool = True
        scale: float = -1.0
        scale_type: str = "static"
        view_dependent_prompting: bool = True
        xs_delta_t: int = 200
        xs_inv_steps: int = 5
        xs_eta: int = 0
        delta_t: int = 80
        delta_t_start: int = 100
        annealing_intervals: bool = False
        warm_up_rate: float = 0.5
        phase: int = 1
        prompt: str = ""
        custom: str = ""
        use_cross: bool = False
        position: int = 0

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            # "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }
        self.controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            controlnet=self.controlnet,
            **pipe_kwargs,
        ).to(self.device)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # del self.pipe.text_encoder
        cleanup()

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        if self.cfg.token_merging:
            import tomesd

            tomesd.apply_patch(self.unet, **self.cfg.token_merging_params)

        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        num_timesteps = self.scheduler.config.num_train_timesteps
        self.scheduler.set_timesteps(num_timesteps, device=self.device)
        self.timesteps = torch.flip(self.scheduler.timesteps, dims=(0, ))
        
        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded Stable Diffusion!")
        
        image_encoder_path = "threestudio/models/guidance/adapter/models/image_encoder/"

        if self.cfg.custom != "":
            self.num_tokens=16
            self.ip_ckpt = self.cfg.custom
        elif self.cfg.use_plus == True:
            self.num_tokens=16
            self.ip_ckpt = "threestudio/models/guidance/adapter/models/ip-adapter-plus_sd15.bin"
            threestudio.info(f"Loaded Plus model!")    
        else:
            self.num_tokens=4
            self.ip_ckpt = "threestudio/models/guidance/adapter/models/ip-adapter_sd15.bin"
            threestudio.info(f"Loaded Vanilla model!")

        self.adapter = IPAdapter(self.pipe, self.ip_ckpt, image_encoder_path, self.cfg.use_cross)
        if self.cfg.use_cross == True:
            self.adapter.set_position(self.cfg.position)
        
        threestudio.info(f"Loaded IP-Adapter!")


    def get_prependicualr_component(self, x, y):
        assert x.shape == y.shape
        return x - ((torch.mul(x, y).sum())/(torch.norm(y)**2)) * y

    def weighted_prependicular_aggricator(self, delta_noise_pred_pos, w_pos, delta_noise_pred_neg, w_neg):
        
        main_positive = delta_noise_pred_pos[0].unsqueeze(0)
        accumulated_output = 0
        for i, complementory_positive in enumerate(delta_noise_pred_pos[1:]):
            accumulated_output +=  w_pos[i] * self.get_prependicualr_component(complementory_positive.unsqueeze(0), main_positive)
            
        for i, w_n in enumerate(w_neg):
            accumulated_output -= w_n * self.get_prependicualr_component(delta_noise_pred_neg[i].unsqueeze(0), main_positive)
            
        return accumulated_output + main_positive

    def set_image(self, img):
        self.image = img
        self.image = self.image.resize((224, 224))
        
    def add_tokens_to_model(
        self, 
        learned_embeds: Mapping[str, Tensor], 
        override_token: Optional[Union[str, dict]] = None
    ) -> None:
    
        # Loop over learned embeddings
        new_tokens = []
        for token, embedding in learned_embeds.items():
            embedding = embedding.to(
                self.text_encoder.get_input_embeddings().weight.dtype)
            if override_token is not None:
                token = override_token if isinstance(
                    override_token, str) else override_token[token]
    
            # Add the token to the tokenizer
            num_added_tokens = self.tokenizer.add_tokens(token)
            if num_added_tokens == 0:
                raise ValueError((f"The tokenizer already contains the token {token}. Please pass a "
                                  "different `token` that is not already in the tokenizer."))
    
            # Resize the token embeddings
            self.text_encoder._resize_token_embeddings(len(self.tokenizer))
    
            # Get the id for the token and assign the embeds
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            self.text_encoder.get_input_embeddings(
            ).weight.data[token_id] = embedding
            new_tokens.append(token)
    
        threestudio.info(
            f'Added {len(new_tokens)} tokens to tokenizer and text embedding: {new_tokens}')

    # def get_text_embeds(self, text):
    
    def add_tokens_to_model_from_path(self, learned_embeds_path: str, override_token: Optional[Union[str, dict]] = None) -> None:
        r"""Loads tokens from a file and adds them to the tokenizer and text encoder of a model."""
        learned_embeds: Mapping[str, Tensor] = torch.load(
            learned_embeds_path, map_location='cpu')
        self.add_tokens_to_model(learned_embeds, override_token)
        
    def add_noise_with_cfg(
        self, 
        latents,
        canny,
        noise,                
        ind_t, 
        ind_prev_t,
        condition_scale,
        cfg=1.0,       
        text_embeddings=None,     
        delta_t=1, 
        inv_steps=1,  
        is_noisy_latent=False,        
        eta=0.0,
    ):
    
        if is_noisy_latent:
            prev_noisy_lat = latents
        else:
            prev_noisy_lat = self.scheduler.add_noise(latents, noise, self.timesteps[ind_prev_t])
        
        cur_ind_t = ind_prev_t
        cur_noisy_lat = prev_noisy_lat

        pred_scores = []
        # latent_model_input = cur_noisy_lat_.repeat(text_embeddings.shape[0], 1, 1, 1).reshape(-1, 4, 64, 64,)
        # timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(text_embeddings.shape[0], 1).reshape(-1)
        for i in range(inv_steps):
            # pred noise
            cur_noisy_lat_ = self.scheduler.scale_model_input(cur_noisy_lat, self.timesteps[cur_ind_t]).to(torch.float16)
            
            if cfg > 1.0:
                latent_model_input = cur_noisy_lat_.repeat(2, 1, 1, 1).reshape(-1, 4, 64, 64,)
                timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
                unet_output = self.unet(
                    latent_model_input, 
                    timestep_model_input, 
                    encoder_hidden_states=text_embeddings
                ).sample
                
                uncond, cond = torch.chunk(unet_output, chunks=2)
                
                unet_output = cond + cfg * (uncond - cond) # reverse cfg to enhance the distillation
            else:
                timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(cur_noisy_lat_.shape[0], 1).reshape(-1)
                
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    cur_noisy_lat_,
                    timestep_model_input,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=canny,
                    conditioning_scale=0.5,
                    guess_mode=True,
                    return_dict=False,
                )

                
                unet_output = self.unet(
                    cur_noisy_lat_,
                    timestep_model_input,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    encoder_hidden_states=text_embeddings
                ).sample


            pred_scores.append((cur_ind_t, unet_output))

            next_ind_t = min(cur_ind_t + delta_t, ind_t)
            cur_t, next_t = self.timesteps[cur_ind_t].to(self.device), self.timesteps[next_ind_t].to(self.device)
            delta_t_ = next_t-cur_t if isinstance(self.scheduler, DDIMScheduler) else next_ind_t-cur_ind_t

            cur_noisy_lat = ddim_step(self.scheduler, unet_output, cur_t, cur_noisy_lat, -delta_t_, eta).prev_sample
            cur_ind_t = next_ind_t

            del unet_output
            torch.cuda.empty_cache()

            if cur_ind_t == ind_t:
                break

        return prev_noisy_lat, cur_noisy_lat, pred_scores[::-1]
        
    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def compute_grad_sds(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        image: Float[Tensor, "B 3 512 512"],
        canny: Float[Tensor, "B 1 512 512"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        condition_scale: Float
    ):
        batch_size = elevation.shape[0]

        def cosine_schedule(start, end, step):
            # torch.linspace를 사용하여 0부터 pi까지 균등하게 분포된 텐서 생성
            t = torch.linspace(0, torch.pi, 981)
            # 코사인 스케줄 계산
            schedule = end + (start - end) * 0.5 * (1 + torch.cos(t))
            # schedule = end + (start - end) * (1 - torch.sin(t))
            # schedule = end + (start - end) * (torch.sin(t))
            # 해당 step의 값을 반환
            return schedule[step]
        # def cosine_schedule(t):
        #     return (1 - math.cos(2 * math.pi * t / 1000)) / 2
            
        if self.cfg.scale == -1.0:
            # scale = max(int(cosine_schedule(981, 20, t.item()).item()) / 1000, 0.2)
            # print(t.item(), scale)
            # scale = (1 - abs(azimuth[1].item()) / 180) * max(1 - t.item() / 1000, 0.2)
            scale = max(1 - t.item() / 1000, 0.4)
        else:
            if self.cfg.scale_type == "linear":
                scale = (1 - abs(azimuth[1].item()) / 180) * self.cfg.scale
            elif self.cfg.scale_type == "cosine":
                scale = ((torch.cos(torch.deg2rad(torch.abs(azimuth[1]))) + 1) / 2) * self.cfg.scale
            elif self.cfg.scale_type == "polynomial":
                scale = ((1 - (abs(azimuth[1].item()) / 180) ** 3)) * self.cfg.scale
            else:
                scale = self.cfg.scale
            
        self.adapter.set_scale(scale)
        
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
        )
        
#         negative_image = Image.effect_mandelbrot((224, 224), (-3, -2.5, 2, 2.5), 100)
        
        cond_image_embeddings, uncond_image_embeddings = self.adapter.get_image_embeds(
            self.image, None, None
        )

        # cond_image_embeddings, uncond_image_embeddings = self.adapter.get_image_embeds(
        #     self.image, None, None
        # )
        if cond_image_embeddings.shape[0] > 1:
            cond_image_embeddings = torch.cat(cond_image_embeddings.chunk(cond_image_embeddings.shape[0]), dim=1)
        if uncond_image_embeddings.shape[0] > 1:
            uncond_image_embeddings = torch.cat(uncond_image_embeddings.chunk(uncond_image_embeddings.shape[0]), dim=1)

        cond_image_embeddings = cond_image_embeddings.repeat(batch_size, 1, 1)
        uncond_image_embeddings = uncond_image_embeddings.repeat(batch_size, 1, 1)

        cond_text_embeddings, uncond_text_embeddings = text_embeddings.chunk(2)
        cond_embeddings = torch.cat([cond_text_embeddings, cond_image_embeddings], dim=1)
        uncond_embeddings = torch.cat([uncond_text_embeddings, uncond_image_embeddings], dim=1)

        if self.cfg.annealing_intervals:
            current_delta_t =  int(self.cfg.delta_t + (self.cfg.warm_up_rate)*(self.cfg.delta_t_start - self.cfg.delta_t))
        else:
            current_delta_t =  self.cfg.delta_t

        neg_guidance_weights = None
        ind_prev_t = max(t - current_delta_t, torch.ones_like(t) * 0)
        starting_ind = max(ind_prev_t - self.cfg.xs_delta_t * self.cfg.xs_inv_steps, torch.ones_like(t) * 0)
        # mu = max(int(t - 0.7 * current_delta_t), torch.ones_like(t) * 0)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.cfg.seed)
        noise = torch.randn(latents.shape, generator=generator, device=self.device)
        
        with torch.no_grad():
            _, prev_latents_noisy, pred_scores_xs = self.add_noise_with_cfg(latents, canny, noise, ind_prev_t, starting_ind, condition_scale,
                                                                        1.0, uncond_embeddings, self.cfg.xs_delta_t, self.cfg.xs_inv_steps, eta=0.0)
            _, latents_noisy, pred_scores_xt = self.add_noise_with_cfg(prev_latents_noisy, canny, noise, t, ind_prev_t, condition_scale,
                                                                    1.0, uncond_embeddings, current_delta_t, 1, eta=0.0, is_noisy_latent=True)
            # _, latents_noisy_mu, pred_scores_mu = self.add_noise_with_cfg(prev_latents_noisy, noise, mu, ind_prev_t,
            #                                                         1.0, uncond_embeddings, current_delta_t, 1, eta=0.0, is_noisy_latent=True)

        pred_scores = pred_scores_xt + pred_scores_xs
        real_noise = pred_scores[0][1]
        
        text_embeddings = torch.cat([cond_embeddings, uncond_embeddings], dim=0)

        # self.adapter.set_scale(scale)
        
        with torch.no_grad():
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                torch.cat([t] * 2 * batch_size),
                encoder_hidden_states=text_embeddings,
                controlnet_cond=torch.cat([canny]*2, dim=0),
                conditioning_scale=0.5,
                guess_mode=True,
                return_dict=False,
            )

            
            noise_pred = self.unet(
                latent_model_input,
                torch.cat([t] * 2 * batch_size),
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                encoder_hidden_states=text_embeddings
            ).sample

        # perform guidance
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        if self.cfg.phase == 1:
            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        else:
            noise_pred = noise_pred_uncond + new_guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        
        rescale = 0.5
        std_pos = noise_pred_text.std([1, 2, 3], keepdim=True)
        std_cfg = noise_pred.std([1, 2, 3], keepdim=True)
        factor = std_pos / std_cfg
        factor = rescale * factor + (1 - rescale)
        noise_pred = noise_pred * factor
        
        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        alpha = (self.alphas[t] ** 0.5).view(-1, 1, 1, 1)
        sigma = ((1 - self.alphas[t]) ** 0.5).view(-1, 1, 1, 1)
        latents_denoised = (latents_noisy - sigma * noise_pred) / alpha
        image_denoised = self.decode_latents(latents_denoised)
        
        grad = w * (noise_pred - real_noise)
        # image-space SDS proposed in HiFA: https://hifa-team.github.io/HiFA-site/
        if self.cfg.use_img_loss:
            grad_img = w * (image - image_denoised) * alpha / sigma
        else:
            grad_img = None

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "noise_pred": noise_pred,
        }

        return grad, grad_img, guidance_eval_utils
        
    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        canny: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        condition_scale: Float,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        guidance_eval=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        canny_BCHW = canny.permute(0, 3, 1, 2)
        
        latents: Float[Tensor, "B 4 64 64"]
        rgb_BCHW_512 = F.interpolate(
            rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
        )
        canny_BCHW_512 = F.interpolate(
            canny_BCHW, (512, 512), mode="bilinear", align_corners=False
        )
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            (1,),
            dtype=torch.long,
            device=self.device,
        )
        # t = kwargs['t_mid']

        grad, grad_img, guidance_eval_utils = self.compute_grad_sds(
            latents,
            rgb_BCHW_512,
            canny_BCHW_512,
            t,
            prompt_utils,
            elevation,
            azimuth,
            camera_distances,
            condition_scale
        )

        grad = torch.nan_to_num(grad)

        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (latents - grad).detach()
        # target = (latents - grad).detach()
        # print(target.shape)
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad

        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        #  / batch size로 나누는 것을 뺌
        # loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        guidance_out = {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

        if self.cfg.use_img_loss:
            grad_img = torch.nan_to_num(grad_img)
            if self.grad_clip_val is not None:
                grad_img = grad_img.clamp(-self.grad_clip_val, self.grad_clip_val)
            target_img = (rgb_BCHW_512 - grad_img).detach()
            loss_sds_img = (
                0.5 * F.mse_loss(rgb_BCHW_512, target_img, reduction="sum") / batch_size
            )
            guidance_out["loss_sds_img"] = loss_sds_img

        if guidance_eval:
            guidance_eval_out = self.guidance_eval(**guidance_eval_utils)
            texts = []
            for n, e, a, c in zip(
                guidance_eval_out["noise_levels"], elevation, azimuth, camera_distances
            ):
                texts.append(
                    f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}"
                )
            guidance_eval_out.update({"texts": texts})
            guidance_out.update({"eval": guidance_eval_out})

        return guidance_out

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_noise_pred(
        self,
        latents_noisy,
        t,
        text_embeddings,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        batch_size = latents_noisy.shape[0]

        if use_perp_neg:
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 4).to(self.device),
                encoder_hidden_states=text_embeddings,
            )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 2).to(self.device),
                encoder_hidden_states=text_embeddings,
            )
            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        return noise_pred

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(
        self,
        t_orig,
        text_embeddings,
        latents_noisy,
        noise_pred,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        # use only 50 timesteps, and find nearest of those to t
        self.scheduler.set_timesteps(50)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)
        bs = (
            min(self.cfg.max_items_eval, latents_noisy.shape[0])
            if self.cfg.max_items_eval > 0
            else latents_noisy.shape[0]
        )  # batch size
        large_enough_idxs = self.scheduler.timesteps_gpu.expand([bs, -1]) > t_orig[
            :bs
        ].unsqueeze(
            -1
        )  # sized [bs,50] > [bs,1]
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t = self.scheduler.timesteps_gpu[idxs]

        fracs = list((t / self.scheduler.config.num_train_timesteps).cpu().numpy())
        imgs_noisy = self.decode_latents(latents_noisy[:bs]).permute(0, 2, 3, 1)

        # get prev latent
        latents_1step = []
        pred_1orig = []
        for b in range(bs):
            step_output = self.scheduler.step(
                noise_pred[b : b + 1], t[b], latents_noisy[b : b + 1], eta=1
            )
            latents_1step.append(step_output["prev_sample"])
            pred_1orig.append(step_output["pred_original_sample"])
        latents_1step = torch.cat(latents_1step)
        pred_1orig = torch.cat(pred_1orig)
        imgs_1step = self.decode_latents(latents_1step).permute(0, 2, 3, 1)
        imgs_1orig = self.decode_latents(pred_1orig).permute(0, 2, 3, 1)

        latents_final = []
        for b, i in enumerate(idxs):
            latents = latents_1step[b : b + 1]
            text_emb = (
                text_embeddings[
                    [b, b + len(idxs), b + 2 * len(idxs), b + 3 * len(idxs)], ...
                ]
                if use_perp_neg
                else text_embeddings[[b, b + len(idxs)], ...]
            )
            neg_guid = neg_guidance_weights[b : b + 1] if use_perp_neg else None
            for t in tqdm(self.scheduler.timesteps[i + 1 :], leave=False):
                # pred noise
                noise_pred = self.get_noise_pred(
                    latents, t, text_emb, use_perp_neg, neg_guid
                )
                # get prev latent
                latents = self.scheduler.step(noise_pred, t, latents, eta=1)[
                    "prev_sample"
                ]
            latents_final.append(latents)

        latents_final = torch.cat(latents_final)
        imgs_final = self.decode_latents(latents_final).permute(0, 2, 3, 1)

        return {
            "bs": bs,
            "noise_levels": fracs,
            "imgs_noisy": imgs_noisy,
            "imgs_1step": imgs_1step,
            "imgs_1orig": imgs_1orig,
            "imgs_final": imgs_final,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        if self.cfg.sqrt_anneal:
            percentage = (
                float(global_step) / self.cfg.trainer_max_steps
            ) ** 0.5  # progress percentage
            if type(self.cfg.max_step_percent) not in [float, int]:
                max_step_percent = self.cfg.max_step_percent[1]
            else:
                max_step_percent = self.cfg.max_step_percent
            curr_percent = (
                max_step_percent - C(self.cfg.min_step_percent, epoch, global_step)
            ) * (1 - percentage) + C(self.cfg.min_step_percent, epoch, global_step)
            self.set_min_max_steps(
                min_step_percent=curr_percent,
                max_step_percent=curr_percent,
            )
        else:
            self.set_min_max_steps(
                min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
                max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
            )

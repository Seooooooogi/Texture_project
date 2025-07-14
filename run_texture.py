import os
import threestudio

import math
from tqdm import tqdm
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import kiui

from transformers import AutoTokenizer, CLIPTextModel
from controlnet_aux import CannyDetector

from threestudio.models.materials.pbr_material import PBRMaterial
from threestudio.models.geometry.custom_mesh import CustomMesh
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.data.uncond import RandomCameraDataModuleConfig, RandomCameraDataset, RandomCameraIterableDataset, RandomCameraDataModule
from threestudio.models.background.solid_color_background import SolidColorBackground
from threestudio.models.renderers.nvdiff_rasterizer import NVDiffRasterizer
from threestudio.utils.saving import SaverMixin
from threestudio.models.exporters.mesh_exporter import MeshExporter

def run(obj_path, prompt, img):
    
    negative_prompt = "oversaturated color, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, poorly drawn eyes, low contrast, underexposed, overexposed, bright blue spots, glowing blue patches, intense blue highlights, Unrealistic highlights, Artificial shininess, Exaggerated light reflections, Unnatural facial expression, Inauthentic eye contact, low resolution"
    tgt_prompt = ""
    
    config = {
        'material_activation': 'sigmoid',
        'environment_texture': "load/lights/mud_road_puresky_1k.hdr",
        'environment_scale': 2.0,
        'min_metallic': 0.0,
        'max_metallic': 0.9,
        'min_roughness': 0.08,
        'max_roughness': 0.9,
        'use_bump': False
    }
    
    background_config = {
    }
    
    camera_config = {
      'height': 512,
      'batch_size': 1,
      'width': 512,
      'camera_distance_range': [3.5, 3.5],
      'fovy_range': [25, 45],
      'camera_perturb': 0.,
      'center_perturb': 0.,
      'up_perturb': 0.,
      'elevation_range': [-20, 45],
      'azimuth_range': [-180, 180],
      'batch_uniform_azimuth': True,
      'eval_camera_distance': 3.5,
      'eval_fovy_deg': 45.,
    }
    
    rasterizer_config = {
        'context_type': "cuda",
    }
    
    config_mesh = {
        'shape_init': f"mesh:{obj_path}",
        'shape_init_params': 1.0,
        'radius': 1.0, # consistent with coarse
        'pos_encoding_config': {
            'otype': 'HashGrid',
            'n_levels': 16,
            'n_features_per_level': 2,
            'log2_hashmap_size': 19,
            'base_resolution': 16,
            'per_level_scale': 1.4472692374403782 # max resolution 4096
        },
        # 'shape_init_mesh_up': '-z',
        # 'shape_init_mesh_front': '+y',
        'shape_init_mesh_up': '+y',
        'shape_init_mesh_front': '+z',
        'n_feature_dims': 8 # albedo3 + roughness1 + metallic1 + bump3
    }
    
    material = PBRMaterial(config)
    mesh = CustomMesh(config_mesh)
    rasterizer = NVDiffRasterizerContext("cuda", "cuda")
    background = SolidColorBackground(background_config)
    detector = CannyDetector()
    
    # CUDA 설정
    mesh = mesh.to("cuda")
    material = material.to("cuda")
    background = background.to("cuda")

    rasterizer = NVDiffRasterizer(rasterizer_config, mesh, material, background)
    testloader = RandomCameraDataModule(camera_config)
    testloader.setup("test")
    
    tokenizer = AutoTokenizer.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="tokenizer"
    )
    
    text_encoder = CLIPTextModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="text_encoder",
        device_map="auto",
    )
    
    with torch.no_grad():
        tokens = tokenizer(
            prompt+", front view",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
    
        special_tokens = tokenizer(
            "front",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        
        positions = [
            i for i, token in enumerate(tokens['input_ids'][0]) 
            if token in special_tokens['input_ids'][0] and token != 49407
        ]

    diffusion_config = {
        'max_iters': 3000,
        'seed': 0,
        'scheduler': 'cosine',
        'mode': 'latent',
        'prompt_processor_type': 'stable-diffusion-prompt-processor',
        'prompt_processor': {
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'use_perp_neg': False,
        },
        'guidance_type': 'stable-diffusion-ip-guidance',
        'guidance': {
            'half_precision_weights': True,
            'pretrained_model_name_or_path': 'runwayml/stable-diffusion-v1-5',
            'guidance_scale': 7.5,
            'weighting_strategy': 'fantasia3d',
            'min_step_percent': 0.02,
            'max_step_percent': 0.98,
            'grad_clip': None,
            'view_dependent_prompting': True,
            'use_plus': True,
            'prompt': prompt,
            'seed': 0,
            'xs_delta_t': 200,
            'xs_inv_steps': 5,
            'xs_eta': 0,
            'delta_t': 50,
            'delta_t_start': 100,
            'annealing_intervals': True,
            'use_img_loss': False,
            'position': positions[1],
    
            'scale_type': "static",
            'use_cross': False
        },
        'image': {
            'width': 512,
            'height': 512,
        }
    }
    guidance = None
    prompt_processor = None
    
    with torch.no_grad():
        torch.cuda.empty_cache()

    guidance = threestudio.find(diffusion_config['guidance_type'])(diffusion_config['guidance'])
    prompt_processor = threestudio.find(diffusion_config['prompt_processor_type'])(diffusion_config['prompt_processor'])
    prompt_processor.configure_text_encoder()

    encoding_optimizer = torch.optim.AdamW(list(mesh.encoding.parameters()), lr=0.01, betas=(0.9, 0.99), eps=1e-15)
    optimizer = torch.optim.AdamW(list(mesh.feature_network.parameters()), lr=0.001, betas=(0.9, 0.99), eps=1e-15)
    num_steps = diffusion_config['max_iters']

    guidance.set_image(img)
    
    def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
            
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, int(num_steps*1.5)) if diffusion_config['scheduler'] == 'cosine' else None

    scaler1 = GradScaler()
    scaler2 = GradScaler()
    
    seed = guidance.cfg.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    camera_module_config = RandomCameraDataModuleConfig(**camera_config)
    dataset = RandomCameraIterableDataset(camera_module_config)
    
    for step in tqdm(range(num_steps + 1)):
        guidance.update_step(epoch=step, global_step=step)
        encoding_optimizer.zero_grad()
        optimizer.zero_grad()
        # if step % 50 == 0:
        #     dataset.update_step(epoch=step, global_step=step)
        images = []
        normals = []
        cannies = []
    
        resolution = 512
            
        with autocast(enabled=True):
            # for focal graph
            # focal_out = rasterizer(focal_data['mvp_mtx'][0:1].cuda(), focal_data['camera_positions'][0:1].cuda(), focal_data['light_positions'][0:1].cuda(), resolution, resolution, focal_data['c2w'][0:1].cuda())
            # images.append(focal_out['comp_rgb'])
            # cannies.append(torch.tensor(np.array(detector(focal_out['comp_rgb'].squeeze(0).detach().cpu() * 255)) / 255).unsqueeze(0).to(torch.float16).cuda())
            # normals.append(focal_out['cam_normal'].reshape(1, resolution, resolution, 3))
        
            # batch = {
            #     'elevation': focal_data['elevation'],
            #     'azimuth': focal_data['azimuth'],
            #     'camera_distances': focal_data['camera_distances']
            # }
            
            for i in range(1):
                sample_data = dataset.collate(dataset.batch_size)
                batch = {
                    'elevation': sample_data['elevation'],
                    'azimuth': sample_data['azimuth'],
                    'camera_distances': sample_data['camera_distances']
                }
                # batch['elevation'] = torch.cat([batch['elevation'], sample_data['elevation']], dim=0)
                # batch['azimuth'] = torch.cat([batch['azimuth'], sample_data['azimuth']], dim=0)
                # batch['camera_distances'] = torch.cat([batch['camera_distances'], sample_data['camera_distances']], dim=0)
                out = rasterizer(sample_data['mvp_mtx'][0:1].cuda(), sample_data['camera_positions'][0:1].cuda(), sample_data['light_positions'][0:1].cuda(), resolution, resolution, sample_data['c2w'][0:1].cuda())
                images.append(out['comp_rgb'])
                cannies.append(torch.tensor(np.array(detector(out['comp_rgb'].squeeze(0).detach().cpu() * 255)) / 255).unsqueeze(0).to(torch.float16).cuda())
                normals.append(out['cam_normal'].reshape(1, resolution, resolution, 3))
            if step % 100 == 0:
                for j, sample_data in enumerate(testloader.test_dataloader()):
                    if j == 0 or j == 30 or j == 60 or j == 90:
                        i = 0
                        test_out = rasterizer(sample_data['mvp_mtx'][i:i+1].cuda(), sample_data['camera_positions'][i:i+1].cuda(), sample_data['light_positions'][i:i+1].cuda(), 512, 512, sample_data['c2w'][i:i+1].cuda())
                        # kiui.utils.write_image(f'./test_plot/0/{step}/rgb_{j:03d}.png', test_out['comp_rgb'])
    
            guidance.cfg.warm_up_rate = 1. - min(step/1000, 1.)
            # guidance.cfg.scale = (step / num_steps)
            loss = guidance(torch.cat(images, dim=0), torch.cat(cannies, dim=0), prompt_processor(), 1.0, **batch, rgb_as_latents=False, guidance_eval=False)
            grad = loss['loss_sds']
            
        scaler1.scale(grad).backward(retain_graph=True)
        scaler1.unscale_(encoding_optimizer)
        scaler1.step(encoding_optimizer)
        scaler1.update()
    
        scaler2.scale(grad).backward()
        scaler2.unscale_(optimizer)
        scaler2.step(optimizer)
        scaler2.update()
    
    export_config = {
            'fmt': "obj-mtl",  # in ['obj-mtl', 'obj'], TODO: fbx
            'save_name': "mesh",
            'save_normal': True,
            'save_uv': True,
            'save_texture': True,
            'texture_size': 1024,
            'texture_format': "png",
            # xatlas_chart_options: field(default_factory=dict)
            # xatlas_pack_options: field(default_factory=dict)
            'context_type': "cuda",
    }
    
    exporter = MeshExporter(export_config, mesh, material, background)
    
    output = exporter()
    save_utils = SaverMixin()
    save_utils.set_save_dir("./output/")
    
    for out in output:
        func_name = f"save_{out.save_type}"
        save_func = getattr(save_utils, func_name)
        save_func(f"{out.save_name}", **out.params)
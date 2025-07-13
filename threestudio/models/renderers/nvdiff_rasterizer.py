from dataclasses import dataclass

import nerfacc
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer, VolumeRenderer
from threestudio.utils.misc import get_device
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *


@threestudio.register("nvdiff-rasterizer")
class NVDiffRasterizer(Rasterizer):
    @dataclass
    class Config(VolumeRenderer.Config):
        context_type: str = "gl"

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, get_device())

    def forward(
        self,
        mvp_mtx: Float[Tensor, "B 4 4"],
        camera_positions: Float[Tensor, "B 3"],
        light_positions: Float[Tensor, "B 3"],
        height: int,
        width: int,
        c2w: Float[Tensor, "B 4 4"],
        render_rgb: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        batch_size = mvp_mtx.shape[0]
        mesh = self.geometry.isosurface()

        v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
            mesh.v_pos, mvp_mtx
        )
        rast, _ = self.ctx.rasterize(v_pos_clip, mesh.t_pos_idx, (height, width))
        mask = rast[..., 3:] > 0
        mask_aa = self.ctx.antialias(mask.float(), rast, v_pos_clip, mesh.t_pos_idx)

        # depth rendering
        # depth = v_pos_clip[..., 2:3]
        # depth = torch.tensor([[[(z_val/1)] for z_val in depth.squeeze()]], dtype=torch.float32).to("cuda")
        # depth, _ = self.ctx.interpolate(depth, rast, mesh.t_pos_idx)

        # old_min = torch.min(depth)
        # old_max = torch.max(depth)
        
        # # Define new minimum and maximum depth values (for normalization)
        # new_min = 0
        # new_max = 1
        
        # # Normalize the output tensor to range [new_min, new_max]
        # depth = (depth - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
        # depth = depth.repeat(1, 1, 1, 3)
        
        # out = {"opacity": mask_aa, "mesh": mesh, "depth": depth}
        out = {"opacity": mask_aa, "mesh": mesh}

        gb_normal, _ = self.ctx.interpolate_one(mesh.v_nrm, rast, mesh.t_pos_idx)
        red_channel = gb_normal[:, :, :, 0]

        # Red 채널을 x축(너비, 2번 차원) 방향으로 뒤집기
        red_channel_flipped = 1.0 - red_channel
        
        # 뒤집은 Red 채널을 원래 텐서에 다시 삽입
        gb_normal_flipped = gb_normal.clone()
        gb_normal_flipped[:, :, :, 0] = red_channel_flipped
        
        batched_normal = gb_normal.view(batch_size, -1,3)
        w2c = c2w[:, :3, :3].inverse()
        cam_normal = torch.matmul(w2c, batched_normal.transpose(1,2)).transpose(1,2)
        cam_normal = cam_normal * mask_aa.view(batch_size, -1, 1)
        cam_normal = cam_normal[0] # only visualize one image
        cam_normal[(cam_normal.sum(dim=1) == 0), 2] = 1
        cam_normal = torch.nn.functional.normalize(cam_normal,dim=1) # (-1,1)
        cam_normal_image = ((cam_normal+1.0)/2.0).clip(0, 1)

        gb_normal = F.normalize(gb_normal, dim=-1)
        gb_normal_aa = torch.lerp(
            torch.zeros_like(gb_normal), (gb_normal + 1.0) / 2.0, mask.float()
        )
        gb_normal_aa = self.ctx.antialias(
            gb_normal_aa, rast, v_pos_clip, mesh.t_pos_idx
        )
        out.update({"comp_normal": gb_normal_aa, "cam_normal": cam_normal_image})  # in [0, 1]

        # TODO: make it clear whether to compute the normal, now we compute it in all cases
        # consider using: require_normal_computation = render_normal or (render_rgb and material.requires_normal)
        # or
        # render_normal = render_normal or (render_rgb and material.requires_normal)

        if render_rgb:
            selector = mask[..., 0]

            gb_pos, _ = self.ctx.interpolate_one(mesh.v_pos, rast, mesh.t_pos_idx)
            gb_viewdirs = F.normalize(
                gb_pos - camera_positions[:, None, None, :], dim=-1
            )
            gb_light_positions = light_positions[:, None, None, :].expand(
                -1, height, width, -1
            )

            positions = gb_pos[selector]
            geo_out = self.geometry(positions, output_normal=False)

            extra_geo_info = {}
            if self.material.requires_normal:
                extra_geo_info["shading_normal"] = gb_normal[selector]
            if self.material.requires_tangent:
                gb_tangent, _ = self.ctx.interpolate_one(
                    mesh.v_tng, rast, mesh.t_pos_idx
                )
                gb_tangent = F.normalize(gb_tangent, dim=-1)
                extra_geo_info["tangent"] = gb_tangent[selector]

            rgb_fg = self.material(
                viewdirs=gb_viewdirs[selector],
                positions=positions,
                light_positions=gb_light_positions[selector],
                **extra_geo_info,
                **geo_out
            )
            gb_rgb_fg = torch.zeros(batch_size, height, width, 3).to(rgb_fg)
            gb_rgb_fg[selector] = rgb_fg

            gb_rgb_bg = self.background(dirs=gb_viewdirs)
            gb_rgb = torch.lerp(gb_rgb_bg, gb_rgb_fg, mask.float())
            gb_rgb_aa = self.ctx.antialias(gb_rgb, rast, v_pos_clip, mesh.t_pos_idx)

            out.update({"comp_rgb": gb_rgb_aa, "comp_rgb_bg": gb_rgb_bg})

        return out

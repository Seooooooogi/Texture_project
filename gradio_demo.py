import os
import gradio as gr
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import pyvista as pv
from run_texture import run

ASSETS_DIR      = "./output/"
DEFAULT_OBJ_PATH = os.path.join(ASSETS_DIR, "mesh.obj")

def plot_obj(obj_path, kd_path=None, metal_path=None, rough_path=None):
    mesh = pv.read(obj_path)
    verts = mesh.points                            # (N, 3)
    faces = mesh.faces.reshape(-1, 4)[:, 1:4]      # (M, 3)

    uvs = mesh.active_t_coords                     # (N, 2), 없으면 None

    if kd_path and os.path.exists(kd_path) and uvs is not None:
        img      = Image.open(kd_path).convert("RGB")
        W, H     = img.size
        arr      = np.array(img)
        u_pix    = np.clip((uvs[:,0] * (W - 1)).astype(int), 0, W-1)
        v_pix    = np.clip(((1-uvs[:,1]) * (H - 1)).astype(int), 0, H-1)
        colors   = arr[v_pix, u_pix]               # (N, 3)
        vcolors  = [f"rgb({r},{g},{b})" for r,g,b in colors]
        mesh3d = go.Mesh3d(
            x=verts[:,0], y=verts[:,1], z=verts[:,2],
            i=faces[:,0], j=faces[:,1], k=faces[:,2],
            vertexcolor=vcolors,
            flatshading=True
        )
    else:
        mesh3d = go.Mesh3d(
            x=verts[:,0], y=verts[:,1], z=verts[:,2],
            i=faces[:,0], j=faces[:,1], k=faces[:,2],
            color="white", opacity=1.0, flatshading=True
        )

    fig = go.Figure(mesh3d)
    fig.update_layout(
        scene=dict(
            bgcolor="black",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(eye=dict(x=0, y=0, z=2), up=dict(x=0, y=1, z=0))
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    return fig


def run_model(obj_file, prompt, img):
    run(obj_file, prompt, img)
    return plot_obj(
        DEFAULT_OBJ_PATH,
        kd_path   = os.path.join(ASSETS_DIR, "texture_kd.png"),
        metal_path= os.path.join(ASSETS_DIR, "texture_metalic.png"),
        rough_path= os.path.join(ASSETS_DIR, "texture_roughness.png"),
    )

def clear_all():
    return None, None, None, None

with gr.Blocks() as demo:
    gr.Markdown("## 이미지 기반 3D 메시 텍스처 생성")

    with gr.Row():
        with gr.Column():
            obj_input   = gr.File(label="OBJ 파일", file_types=['.obj','.OBJ'])
            view_button = gr.Button("View Model")
        with gr.Column():
            plot_out    = gr.Plot(label="3D View (File Upload)")

    with gr.Row():
        with gr.Column():
            prompt_input   = gr.Textbox(label="텍스트 입력", placeholder="메시의 클래스에 대한 설명을 입력하세요.")
            img_input      = gr.Image(type="pil", label="이미지 입력")
            process_button = gr.Button("Run Model")
            clear_button   = gr.Button("Clear")
        with gr.Column():
            default_plot   = gr.Plot(label="3D View (Output)")

    view_button.click(
        fn=lambda f: plot_obj(
            f.name if f else DEFAULT_OBJ_PATH,
            kd_path=None, metal_path=None, rough_path=None
        ),
        inputs=[obj_input],
        outputs=[plot_out]
    )
    process_button.click(
        fn=run_model,
        inputs=[obj_input, prompt_input, img_input],
        outputs=[default_plot]
    )
    clear_button.click(
        fn=clear_all,
        inputs=None,
        outputs=[obj_input, prompt_input, img_input, default_plot],
        queue=False
    )

if __name__ == "__main__":
    demo.launch(share=True)
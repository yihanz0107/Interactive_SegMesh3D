import os
import torch
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings, 
    MeshRasterizer, 
    MeshRenderer,
    FoVPerspectiveCameras, 
    AmbientLights,
    PointLights,
    DirectionalLights,
    HardPhongShader,
    SoftPhongShader,
    Materials,
    TexturesVertex,
    Textures,
    
)


def load_and_normalize_mesh(path):
    mesh = trimesh.load(path, force='mesh')

    vertices = mesh.vertices
    centroid = vertices.mean(axis=0)
    vertices -= centroid

    max_dist = np.max(np.linalg.norm(vertices, axis=1))
    if max_dist > 0:
        vertices /= max_dist
        
    mesh.vertices = vertices
    print(f"Mesh loaded and normalized. Vertices: {mesh.vertices.shape}")
    return mesh


def getcamScene(mesh_path, selected_view, use_normal=False):
    device = torch.device("cuda")
    with torch.autocast(device_type="cuda", enabled=False):
        selected_view = torch.tensor(selected_view, dtype=torch.float32, device=device).reshape(4, 4).T
        mesh_normalized = load_and_normalize_mesh(mesh_path)
        verts = torch.tensor(mesh_normalized.vertices, dtype=torch.float32, device=device)
        faces = torch.tensor(mesh_normalized.faces, dtype=torch.int64, device=device)
        selected_view_tensor = torch.tensor(selected_view, dtype=torch.float32, device=device).reshape(4, 4).T
        temp_mesh = Meshes(verts=[verts], faces=[faces])
        verts_normals = temp_mesh.verts_normals_packed()
        lights = None
        shader = None
        materials = None

        if use_normal:
            verts_rgb = (verts_normals + 1.0) * 0.5
            textures = TexturesVertex(verts_features=verts_rgb[None])
            
            lights = AmbientLights(device=device, ambient_color=((1.0, 1.0, 1.0),))
            shader_class = HardPhongShader 

        else:
            gray_color = 0.8 
            verts_rgb = torch.ones_like(verts_normals) * gray_color
            textures = TexturesVertex(verts_features=verts_rgb[None])
            
            cam_pos = selected_view_tensor[:3, 3] # (3,)
            
            light_pos = cam_pos.clone()
            light_pos[0] += -1.0 
            light_pos[1] += 5.0 
            
            lights = PointLights(
                device=device, 
                location=[light_pos.tolist()], 
                ambient_color=((0.7, 0.7, 0.7),),  
                diffuse_color=((0.3, 0.3, 0.3),),
            )
            
            materials = Materials(
                device=device,
                shininess=20.0
            )
            shader_class = SoftPhongShader
        
        pytorch_mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
        H, W = 1024, 1024
        c2w = selected_view.float().to(device)
        w2c = torch.inverse(c2w)
        R_gl = w2c[:3, :3]
        T_gl = w2c[:3, 3]
        convert_matrix = torch.tensor([
            [-1.0, 0.0, 0.0],
            [ 0.0, 1.0, 0.0],
            [ 0.0, 0.0, -1.0]
        ], device=device, dtype=torch.float32)

        R_p3d = convert_matrix @ R_gl
        T_p3d = convert_matrix @ T_gl
        R_final = R_p3d.t().unsqueeze(0)
        T_final = T_p3d.unsqueeze(0)

        cameras = FoVPerspectiveCameras(
            device=device, 
            R=R_final, 
            T=T_final, 
            fov=60.0,
            aspect_ratio=W/H, 
            znear=0.01,
            zfar=16.0
        )


        print("Rendering Normal Map for SAM...")

        raster_settings_rgb = RasterizationSettings(
            image_size=(H, W), 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )

        renderer_rgb = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings_rgb),
            shader=shader_class(
                device=device, 
                cameras=cameras, 
                lights=lights,
                materials=materials
            )
        )

        target_images = renderer_rgb(pytorch_mesh) # (1, H, W, 4)
        rgb_image = target_images[0, ..., :3].cpu().numpy() 
        rgb_image = (rgb_image * 255).astype(np.uint8)
        image_pil = Image.fromarray(rgb_image)

        return image_pil


if __name__=="__main__":

    mesh_path = 'samesh3/assets/jacket.glb' 
    selected_view =[
        -0.9981,  0.0000, -0.0621,  0.0000,
        0.0137,  0.9754, -0.2199,  0.0000,
        0.0606, -0.2203, -0.9736,  0.0000,
        0.1212, -0.4406, -1.9471,  1.0000]

    img = getcamScene(mesh_path, selected_view)
    img.save("./1.png")
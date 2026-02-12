import os
import sys
import numpy as np
import torch
import trimesh
from torch_cluster import nearest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from holopart.pipelines.pipeline_holopart import HoloPartPipeline
from holopart.inference_utils import hierarchical_extract_geometry, flash_extract_geometry
from backend.multiview_glb import enforce_clean_boundary_cut

NUM_SURFACE_SAMPLES = 20480
PART_NORMALIZE_SCALE = 0.7

def close_and_refine_mesh(mesh, smooth_boundary_iters=20):
    """
    Close the open Mesh into a solid object (Watertight).

    Steps:
    1. Smooth the boundary strongly (reuse your existing logic) to make the cuts even.
    2. Identify holes and fill them.
    3. Fix the normal directions to ensure faces are outward-facing.
"""
    print("  - Closing mesh boundaries to make it watertight...")
    mesh = enforce_clean_boundary_cut(mesh, iterations=smooth_boundary_iters, strength=0.8)
    try:
        mesh.fill_holes()
    except Exception as e:
        print(f"    Warning: Standard hole filling failed ({e}). Trying repair lookup...")

    if not mesh.is_watertight:
        print("    Mesh is still open. Attempting aggressive repair...")
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_inversion(mesh)
        
        try:
            trimesh.repair.fill_holes(mesh)
        except:
            pass

    try:
        mesh.fix_normals()
    except:
        pass
        
    print(f"    Final Mesh Status -> Watertight: {mesh.is_watertight}, Faces: {len(mesh.faces)}")
    return mesh



def prepare_data(data_path, device="cuda"):
    if data_path.endswith(".glb"):
        parts_mesh = trimesh.load(data_path)
        part_name_list = []
        part_pcd_list = []
        whole_cond_list = []
        part_cond_list = []
        part_local_cond_list = []
        part_center_list = []
        part_scale_list = []
        for i, (name, part_mesh) in enumerate(parts_mesh.geometry.items()):
            part_surface_points, face_idx = part_mesh.sample(NUM_SURFACE_SAMPLES, return_index=True)
            part_surface_normals = part_mesh.face_normals[face_idx]
            part_pcd = np.concatenate([part_surface_points, np.ones_like(part_surface_points[:, :1])*i], axis=-1)
            part_pcd_list.append(part_pcd)

            part_surface_points = torch.FloatTensor(part_surface_points)
            part_surface_normals = torch.FloatTensor(part_surface_normals)
            part_cond = torch.cat([part_surface_points, part_surface_normals], dim=-1)
            part_local_cond = part_cond.clone()
            part_cond_max = part_local_cond[:, :3].max(dim=0)[0]
            part_cond_min = part_local_cond[:, :3].min(dim=0)[0]
            part_center_new = (part_cond_max + part_cond_min) / 2
            part_local_cond[:, :3] = part_local_cond[:, :3] - part_center_new
            part_scale_new = (part_local_cond[:, :3].abs().max() / (0.95 * PART_NORMALIZE_SCALE)).item()
            part_local_cond[:, :3] = part_local_cond[:, :3] / part_scale_new
            part_cond_list.append(part_cond)
            part_local_cond_list.append(part_local_cond)
            part_name_list.append(name)
            part_center_list.append(part_center_new)
            part_scale_list.append(part_scale_new)
        
        part_pcd = np.concatenate(part_pcd_list, axis=0)
        part_pcd = torch.FloatTensor(part_pcd).to(device)
        whole_mesh = parts_mesh.dump(concatenate=True)
        whole_surface_points, face_idx = whole_mesh.sample(NUM_SURFACE_SAMPLES, return_index=True)
        whole_surface_normals = whole_mesh.face_normals[face_idx]
        whole_surface_points = torch.FloatTensor(whole_surface_points)
        whole_surface_normals = torch.FloatTensor(whole_surface_normals)
        whole_surface_points_tensor = whole_surface_points.to(device)
        nearest_idx = nearest(whole_surface_points_tensor, part_pcd[:, :3])
        nearest_part = part_pcd[nearest_idx]
        nearest_part = nearest_part[:, 3].cpu()
        for i in range(len(part_cond_list)):
            surface_points_part_mask = (nearest_part == i).float()
            whole_cond = torch.cat([whole_surface_points, whole_surface_normals, surface_points_part_mask[..., None]], dim=-1)
            whole_cond_list.append(whole_cond)

        batch_data = {
            "whole_cond": torch.stack(whole_cond_list, dim=0).to(device),
            "part_cond": torch.stack(part_cond_list, dim=0).to(device),
            "part_local_cond": torch.stack(part_local_cond_list, dim=0).to(device),
            "part_id_list": part_name_list,
            "part_center_list": part_center_list,
            "part_scale_list": part_scale_list,
        }
    else:
        raise ValueError("Unsupported file format. Please provide a .glb file.")
    
    return batch_data
        


def extract_original_parts(
    mesh_input: str,
    device: str = "cuda" 
) -> list:
    if not mesh_input.endswith(".glb"):
        raise ValueError("Unsupported file format. Please provide a .glb file.")

    print(f"Loading original parts from: {mesh_input}")
    parts_mesh = trimesh.load(mesh_input)
    
    mesh_list = []
    
    for i, (name, part_mesh) in enumerate(parts_mesh.geometry.items()):
        mesh_copy = part_mesh.copy()
    #     mesh_copy = enforce_clean_boundary_cut(
    #     mesh_copy, 
    #     iterations=50, 
    #     strength=0.9, 
    #     close_hole=True # Watertight
    # )
        
        mesh_copy.metadata['name'] = name
        
        mesh_list.append(mesh_copy)
        
    print(f"Extracted {len(mesh_list)} parts directly from file.")
    return mesh_list

@torch.no_grad()
def run_holopart(
    mesh,
    batch_size =  8,
    seed = 42,
    num_inference_steps: int = 50,
    guidance_scale: float = 3.5,
    dense_octree_depth=8,
    hierarchical_octree_depth=9,
    flash_octree_depth=9,
    final_octree_depth=-1,
    num_chunks=10000,
    use_flash_decoder: bool = True,
    bounds=(-1.05, -1.05, -1.05, 1.05, 1.05, 1.05),
    post_smooth=True,
    device: str = "cuda",
) -> trimesh.Scene:
    batch = prepare_data(mesh, device='cuda')
    part_surface = batch["part_cond"]
    whole_surface = batch["whole_cond"]
    part_local_surface = batch["part_local_cond"]
    part_id_list = batch["part_id_list"]
    part_center_list = batch["part_center_list"]
    part_scale_list = batch["part_scale_list"]

    latent_list = []
    mesh_list = []
    
    pipe = HoloPartPipeline.from_pretrained("holopart/pretrained_weights").to(device='cuda', dtype=torch.float16)

    random_colors = np.random.rand(len(part_surface), 3)

    for i in range(0, len(part_surface), batch_size):
        part_surface_batch = part_surface[i : i + batch_size]
        whole_surface_batch = whole_surface[i : i + batch_size]
        part_local_surface_batch = part_local_surface[i : i + batch_size]

        meshes_latent = pipe(
            part_surface=part_surface_batch,
            whole_surface=whole_surface_batch,
            part_local_surface=part_local_surface_batch,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device=device).manual_seed(seed),
            output_type="latent",
        ).samples
        latent_list.append(meshes_latent)
    meshes_latent = torch.cat(latent_list, dim=0)

    if use_flash_decoder:
        pipe.vae.set_flash_decoder()
    for i, mesh_latent in enumerate(meshes_latent):
        mesh_latent = mesh_latent.unsqueeze(0)
        if use_flash_decoder:
            output = flash_extract_geometry(
                mesh_latent,
                pipe.vae,
                bounds=bounds,
                octree_depth=flash_octree_depth,
                num_chunks=num_chunks,
            )
        else:
            geometric_func = lambda x: pipe.vae.decode(mesh_latent, sampled_points=x).sample
            output = hierarchical_extract_geometry(
                geometric_func,
                device,
                bounds=bounds,
                dense_octree_depth=dense_octree_depth,
                hierarchical_octree_depth=hierarchical_octree_depth,
                final_octree_depth=final_octree_depth,
                post_smooth=post_smooth
            )
        meshes = [trimesh.Trimesh(mesh_v_f[0].astype(np.float32), mesh_v_f[1]) for mesh_v_f in output]
        part_mesh = trimesh.util.concatenate(meshes)
        part_mesh.visual.vertex_colors = random_colors[i]
        part_mesh.name = part_id_list[i]
        part_mesh.apply_scale(part_scale_list[i])
        part_mesh.apply_translation(part_center_list[i])
        mesh_list.append(part_mesh)
    return mesh_list

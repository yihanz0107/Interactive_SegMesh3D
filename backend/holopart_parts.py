import os
import sys
import trimesh

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.multiview_glb import enforce_clean_boundary_cut

NUM_SURFACE_SAMPLES = 20480
PART_NORMALIZE_SCALE = 0.7


def extract_original_parts(
    mesh_input: str,
) -> list:
    if not mesh_input.endswith(".glb"):
        raise ValueError("Unsupported file format. Please provide a .glb file.")

    print(f"Loading original parts from: {mesh_input}")
    parts_mesh = trimesh.load(mesh_input)
    
    mesh_list = []
    
    for i, (name, part_mesh) in enumerate(parts_mesh.geometry.items()):
        mesh_copy = part_mesh.copy()
        mesh_copy.metadata['name'] = name
        mesh_list.append(mesh_copy)
    print(f"Extracted {len(mesh_list)} parts directly from file.")
    return mesh_list

def run_holopart(
    mesh_input: str,
) -> list:
    if not mesh_input.endswith(".glb"):
        raise ValueError("Unsupported file format. Please provide a .glb file.")

    print(f"Loading original parts from: {mesh_input}")
    parts_mesh = trimesh.load(mesh_input)
    
    mesh_list = []
    
    for i, (name, part_mesh) in enumerate(parts_mesh.geometry.items()):
        mesh_copy = part_mesh.copy()
        mesh_copy = enforce_clean_boundary_cut(
        mesh_copy, 
        iterations=50, 
        strength=0.9, 
        close_hole=True # Watertight
    )
        
        mesh_copy.metadata['name'] = name
        
        mesh_list.append(mesh_copy)
        
    print(f"Extracted {len(mesh_list)} parts directly from file.")
    return mesh_list

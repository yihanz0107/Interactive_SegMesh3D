import torch
import trimesh
import numpy as np
import scipy.sparse as sp
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings, 
    MeshRasterizer, 
    FoVPerspectiveCameras
)

import scipy.sparse as sp
from scipy.spatial import cKDTree
import networkx as nx


def enforce_clean_boundary_cut(mesh, iterations=20, strength=0.8, close_hole=True):
    """
    Args:
        close_hole: Whether to automatically fill the boundary after smoothing.
    """
    boundary_groups = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    boundary_edges_indices = mesh.edges_sorted[boundary_groups]
    
    if len(boundary_edges_indices) == 0:
        return mesh
        
    g = nx.Graph()
    g.add_edges_from(boundary_edges_indices)
    vertices = mesh.vertices.copy()
    new_faces = list(mesh.faces.copy())

    for component_nodes in nx.connected_components(g):
        if len(component_nodes) < 3: continue 
        node_list = list(component_nodes)
        subgraph = g.subgraph(node_list)
        
        try:
            ordered_loop_edges = nx.find_cycle(subgraph, source=node_list[0])
            loop_indices = [edge[0] for edge in ordered_loop_edges]
        except nx.NetworkXNoCycle:
            continue
            
        loop_verts = vertices[loop_indices]
        for _ in range(iterations):
            prev_verts = np.roll(loop_verts, 1, axis=0)
            next_verts = np.roll(loop_verts, -1, axis=0)
            target_pos = (prev_verts + next_verts) / 2.0
            loop_verts = loop_verts * (1 - strength) + target_pos * strength
        vertices[loop_indices] = loop_verts

        if close_hole:
            center_pos = loop_verts.mean(axis=0)
            center_idx = len(vertices)
            vertices = np.vstack([vertices, center_pos]) 
            
            for i in range(len(loop_indices)):
                v1 = loop_indices[i]
                v2 = loop_indices[(i + 1) % len(loop_indices)]
                new_faces.append([v1, v2, center_idx])

    new_mesh = trimesh.Trimesh(vertices=vertices, faces=new_faces)
    
    try:
        new_mesh.fix_normals()
        trimesh.smoothing.filter_laplacian(new_mesh, iterations=3)
    except:
        pass
        
    return new_mesh

def absorb_nearby_fragments(mesh, mask, proximity_threshold=0.02, max_fragment_ratio=0.05):
    """
    Spatial Adhesion Logic (Proportional Adaptive Version):
    Check all fragments classified as "background". 
    If they are close enough to the "main body" and their volume is small enough 
    (relative proportion), they will be forcibly reclassified as part of the main body.
    
    Args:
        mesh: Trimesh object
        mask: Current face mask (True indicates the main body)
        proximity_threshold: Distance threshold (normalized unit sphere scale, recommended range 0.02 to 0.05)
        max_fragment_ratio: Only fragments with a face count percentage less than this ratio will be adhered (0.05 = 5%)
                              This prevents large background walls from being absorbed.
    """
    total_faces = len(mesh.faces)
    max_fragment_size = int(total_faces * max_fragment_ratio)
    
    print(f"  - Running proximity absorption (Dist={proximity_threshold}, MaxRatio={max_fragment_ratio:.1%})...")
    print(f"    -> Fragment size limit: {max_fragment_size} faces (out of {total_faces})")

    face_mask_bool = mask.astype(bool)
    
    if not np.any(face_mask_bool):
        return mask

    selected_faces = mesh.faces[face_mask_bool]
    unique_selected_vert_indices = np.unique(selected_faces.flatten())
    selected_vertices = mesh.vertices[unique_selected_vert_indices]
    
    # KD-Tree
    tree = cKDTree(selected_vertices)
    
    inverse_mask = ~face_mask_bool
    adjacency = mesh.face_adjacency
    
    valid_edges_mask = inverse_mask[adjacency[:, 0]] & inverse_mask[adjacency[:, 1]]
    subset_edges = adjacency[valid_edges_mask]
    
    if len(subset_edges) == 0:
        return mask

    rows = subset_edges[:, 0]
    cols = subset_edges[:, 1]
    data = np.ones(len(rows), dtype=int)
    num_faces = len(mesh.faces)
    
    matrix = sp.coo_matrix(
        (np.concatenate([data, data]), 
         (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
        shape=(num_faces, num_faces)
    )
    
    n_components, labels = sp.csgraph.connected_components(matrix, directed=False, return_labels=True)
    
    bg_face_indices = np.where(inverse_mask)[0]
    bg_labels = labels[bg_face_indices]
    
    unique_labels, counts = np.unique(bg_labels, return_counts=True)
    
    absorbed_count = 0
    fragments_absorbed = 0
    
    for label, count in zip(unique_labels, counts):
        if count > max_fragment_size:
            continue
            
        current_fragment_mask = (labels == label) & inverse_mask
        fragment_faces = mesh.faces[current_fragment_mask]
        
        fragment_vert_indices = np.unique(fragment_faces.flatten())
        if len(fragment_vert_indices) > 100:
            sample_indices = np.random.choice(fragment_vert_indices, 100, replace=False)
            fragment_vertices = mesh.vertices[sample_indices]
        else:
            fragment_vertices = mesh.vertices[fragment_vert_indices]
        
        dists, _ = tree.query(fragment_vertices, k=1)
        min_dist = np.min(dists)
        
        if min_dist < proximity_threshold:
            mask[current_fragment_mask] = True
            absorbed_count += count
            fragments_absorbed += 1
            
    print(f"    Absorbed {fragments_absorbed} fragments ({absorbed_count} faces) into main mesh.")
    
    return mask



def load_and_normalize_mesh(path):
    mesh = trimesh.load(path, force='mesh')
    vertices = mesh.vertices
    centroid = vertices.mean(axis=0)
    vertices -= centroid
    max_dist = np.max(np.linalg.norm(vertices, axis=1))
    if max_dist > 0:
        vertices /= max_dist
    mesh.vertices = vertices
    return mesh

def build_adjacency_matrix(mesh, device):
    """
    Construct the Face Adjacency Matrix for Diffusion Smoothing
    """
    adjacency = mesh.face_adjacency
    
    # Construct a bidirectional graph
    i = np.concatenate([adjacency[:, 0], adjacency[:, 1]])
    j = np.concatenate([adjacency[:, 1], adjacency[:, 0]])
    
    # Self-loops
    num_faces = len(mesh.faces)
    self_loop = np.arange(num_faces)
    i = np.concatenate([i, self_loop])
    j = np.concatenate([j, self_loop])
    
    indices = torch.tensor(np.stack([i, j]), device=device, dtype=torch.long)
    values = torch.ones(len(i), device=device, dtype=torch.float32)
    
    adj_mat = torch.sparse_coo_tensor(indices, values, (num_faces, num_faces))
    
    degree = torch.sparse.sum(adj_mat, dim=1).to_dense()
    degree[degree == 0] = 1.0 
    
    return adj_mat, degree.unsqueeze(1)

def refine_face_scores(mesh, scores, device, smoothing_iters=10, lambda_smooth=0.5):
    print(f"  - Smoothing scores ({smoothing_iters} iters, lambda={lambda_smooth})...")
    num_faces = len(mesh.faces)
    scores = scores.to(dtype=torch.float32, device=device)
    
    adj_mat, degree = build_adjacency_matrix(mesh, device)
    refined_scores = scores.clone().unsqueeze(1)
    
    with torch.autocast(device_type="cuda", enabled=False):
        for _ in range(smoothing_iters):
            neighbor_sum = torch.sparse.mm(adj_mat, refined_scores)
            neighbor_avg = neighbor_sum / degree
            refined_scores = (1 - lambda_smooth) * refined_scores + lambda_smooth * neighbor_avg
        
    return refined_scores.squeeze()

def apply_morphological_cleanup(mesh, mask, min_island=100, min_hole=100):
    """
    Topology-Based Morphological Processing for Mesh:
        Filling Holes: Convert small False regions into True.
        Removing Islands: Convert small True regions into False.

    Args:
        mesh: Trimesh object
        mask: (N,) bool numpy array, currently selected faces
        min_island: Regions with fewer than this number of selected faces will be removed (denoising)
        min_hole: Regions with fewer than this number of unselected faces will be filled (filling holes)
    """
    print("  - Running topological cleanup (Hole Filling & Island Removal)...")
    
    num_faces = len(mesh.faces)
    adjacency = mesh.face_adjacency
    
    def get_connected_components(subset_mask):
        valid_edges_mask = subset_mask[adjacency[:, 0]] & subset_mask[adjacency[:, 1]]
        subset_edges = adjacency[valid_edges_mask]
        if len(subset_edges) == 0:
            return np.array([]), np.array([])

        rows = subset_edges[:, 0]
        cols = subset_edges[:, 1]
        data = np.ones(len(rows), dtype=int)

        matrix = sp.coo_matrix(
            (np.concatenate([data, data]), 
             (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
            shape=(num_faces, num_faces)
        )

        n_components, labels = sp.csgraph.connected_components(matrix, directed=False, return_labels=True)
        return labels, n_components

    refined_mask = mask.copy()
    inverse_mask = ~refined_mask
    
    if np.any(inverse_mask):
        labels, n_comps = get_connected_components(inverse_mask)
        
        if n_comps > 0:
            inv_indices = np.where(inverse_mask)[0]
            if len(inv_indices) > 0:
                comp_labels = labels[inv_indices]
                unique_labels, counts = np.unique(comp_labels, return_counts=True)

                small_hole_labels = unique_labels[counts < min_hole]
                
                if len(small_hole_labels) > 0:
                    print(f"    Found {len(small_hole_labels)} small holes to fill.")
                    is_small_hole = np.isin(labels, small_hole_labels)
                    faces_to_fill = np.where(is_small_hole & inverse_mask)[0]
                    refined_mask[faces_to_fill] = True

    if np.any(refined_mask):
        labels, n_comps = get_connected_components(refined_mask)
        
        if n_comps > 0:
            true_indices = np.where(refined_mask)[0]
            if len(true_indices) > 0:
                comp_labels = labels[true_indices]
                unique_labels, counts = np.unique(comp_labels, return_counts=True)
                
                small_island_labels = unique_labels[counts < min_island]
                
                if len(small_island_labels) > 0:
                    print(f"    Found {len(small_island_labels)} small islands to remove.")
                    is_small_island = np.isin(labels, small_island_labels)
                    faces_to_remove = np.where(is_small_island & refined_mask)[0]
                    refined_mask[faces_to_remove] = False
    
    return refined_mask

def multiview_voting_to_glb(
    glb_path: str, 
    views_data: list, 
    output_path: str, 
    vote_threshold: float = 0.5,
    device: str = "cuda",
    smoothing_iters: int = 8,   
    min_hole_size: int = 200,   
    min_cluster_size: int = 200  
):
    
    device = torch.device(device)
    print(f"Loading mesh from {glb_path}...")
    
    mesh_normalized = load_and_normalize_mesh(glb_path)
    
    verts = torch.tensor(mesh_normalized.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh_normalized.faces, dtype=torch.int64, device=device)
    pytorch_mesh = Meshes(verts=[verts], faces=[faces])
    
    num_faces = len(mesh_normalized.faces)
    global_total_counts = torch.zeros(num_faces, device=device, dtype=torch.float32)
    global_masked_counts = torch.zeros(num_faces, device=device, dtype=torch.float32)
    
    convert_matrix = torch.tensor([
        [-1.0, 0.0, 0.0],
        [ 0.0, 1.0, 0.0],
        [ 0.0, 0.0, -1.0]
    ], device=device, dtype=torch.float32)

    print(f"Start processing {len(views_data)} views...")

    with torch.autocast(device_type="cuda", enabled=False):
        for idx, view in enumerate(views_data):
            matrix_raw = view["matrix"]
            mask_raw = view["mask"]
            H, W = mask_raw.shape[:2]
            
            if isinstance(matrix_raw, list):
                matrix_raw = np.array(matrix_raw)
            selected_view = torch.tensor(matrix_raw, dtype=torch.float32).reshape(4, 4).T
            
            c2w = selected_view.to(device)
            w2c = torch.inverse(c2w)
            
            R_gl = w2c[:3, :3]
            T_gl = w2c[:3, 3]
            R_p3d = convert_matrix @ R_gl
            T_p3d = convert_matrix @ T_gl
            
            cameras = FoVPerspectiveCameras(
                device=device, R=R_p3d.t().unsqueeze(0), T=T_p3d.unsqueeze(0), 
                fov=60.0, aspect_ratio=W/H, znear=0.01, zfar=16.0
            )
            
            raster_settings = RasterizationSettings(
                image_size=(H, W), blur_radius=0.0, faces_per_pixel=1
            )
            
            rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
            fragments = rasterizer(pytorch_mesh)
            pix_to_face_flat = fragments.pix_to_face.squeeze().flatten()
            
            if isinstance(mask_raw, np.ndarray):
                mask_tensor = torch.from_numpy(mask_raw).to(device)
            else:
                mask_tensor = mask_raw.to(device)
            mask_flat = mask_tensor.flatten().float()
            
            valid_pixels = (pix_to_face_flat >= 0)
            visible_face_ids = pix_to_face_flat[valid_pixels]
            visible_mask_vals = mask_flat[valid_pixels]
            
            global_total_counts.scatter_add_(0, visible_face_ids, torch.ones_like(visible_mask_vals))
            global_masked_counts.scatter_add_(0, visible_face_ids, visible_mask_vals)
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(views_data)} views...")

    print("Calculating initial votes...")
    pos_votes = global_masked_counts
    neg_votes = global_total_counts - global_masked_counts 
    
    # 2. Define Trust Coefficient (Alpha)
        # alpha = 1.0: Equivalent to the original logic (average value)
        # alpha = 2.0: 1 positive vote offsets 2 negative votes (1 positive 2 negative => 0.5 score)
        # alpha = 3.0: 1 positive vote offsets 3 negative votes (more aggressive, suitable for recovering details)
    positive_bias_weight = 4
    
    # 3. Score = (Pos * W) / (Pos * W + Neg)
    weighted_numerator = pos_votes * positive_bias_weight
    weighted_denominator = weighted_numerator + neg_votes + 1e-6
    raw_ratios = weighted_numerator / weighted_denominator
    
    refined_ratios = refine_face_scores(
        mesh_normalized, 
        raw_ratios, 
        device=device, 
        smoothing_iters=smoothing_iters,
        lambda_smooth=0.6 
    )
    
    is_selected_tensor = refined_ratios > vote_threshold
    mask_bool = is_selected_tensor.cpu().numpy() 

    mask_bool = apply_morphological_cleanup(
        mesh_normalized, 
        mask_bool, 
        min_island=0,      
        min_hole=min_hole_size 
    )
    
    mask_bool = absorb_nearby_fragments(
        mesh_normalized,
        mask_bool,
        proximity_threshold=0.04,  # If the distance is less than 0.04 (unit length), it's considered part of the same object
        max_fragment_ratio=0.08     # Only absorb fragments that are less than 8% of the total faces
    )

    # 3. Remove Islands
    mask_bool = apply_morphological_cleanup(
        mesh_normalized, 
        mask_bool, 
        min_island=min_cluster_size,
        min_hole=0 
    )

    selected_face_ids = np.where(mask_bool)[0]

    print(f"Splitting mesh... Selected {len(selected_face_ids)} faces out of {num_faces}.")

    all_indices = np.arange(num_faces)
    unselected_face_ids = np.setdiff1d(all_indices, selected_face_ids)
    
    scene_parts = []
    
    if len(selected_face_ids) > 0:
        part_target = mesh_normalized.submesh([selected_face_ids], append=True)
        part_target.metadata['name'] = 'part_target'
        part_target.visual.vertex_colors = [255, 0, 0, 255]
        scene_parts.append(part_target)
        
    if len(unselected_face_ids) > 0:
        part_background = mesh_normalized.submesh([unselected_face_ids], append=True)
        part_background.metadata['name'] = 'part_background'
        part_background.visual.vertex_colors = [200, 200, 200, 255]
        scene_parts.append(part_background)
    
    if len(scene_parts) > 0:
        scene = trimesh.Scene(scene_parts)
        scene.export(output_path)
        print(f"Saved split scene to: {output_path}")
        return scene
    else:
        print("Error: Resulting mesh is empty.")
        return None
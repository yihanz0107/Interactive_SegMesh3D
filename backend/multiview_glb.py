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
# from scipy.spatial import cKDTree
import networkx as nx
from pytorch3d.ops import knn_points
import pandas
import random
import colorsys

_sysrand = random.SystemRandom()

def get_random_pastel_color():
    """random (RGBA)"""
    h = _sysrand.random()             
    s = _sysrand.uniform(0.5, 1)    
    v = _sysrand.uniform(0.8, 0.9)    
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return [int(r*255), int(g*255), int(b*255), 255]

def absorb_blind_spots_gpu(mesh, mask, sigma=0.05, k_neighbors=30, threshold=0.5, batch_size=10000, device='cuda'):
    """
    [GPU 极速版] 空间高斯吸收 - 使用 PyTorch3D 加速并在显存中计算。
    包含分批处理逻辑，防止显存溢出 (OOM)。
    
    Args:
        mesh: trimesh 对象
        mask: 当前的 bool mask (numpy array)
        sigma: 高斯核带宽
        k_neighbors: 最近邻数量
        threshold: 选中阈值
        batch_size: 每次处理多少个未选中的面 (显存越小，这个数设越小，建议 10000 - 50000)
        device: 'cuda'
    """
    print(f"  - Running GPU spatial Gaussian absorption (Sigma={sigma}, K={k_neighbors})...")
    
    # 1. 准备数据：将面中心转为 Tensor 并移至 GPU
    # 注意：我们假设 mesh 已经归一化，坐标在 [-0.5, 0.5] 或 [-1, 1] 之间
    face_centers_np = mesh.triangles_center
    all_centers = torch.tensor(face_centers_np, dtype=torch.float32, device=device)
    
    # 将 mask 转为 float tensor (0.0 / 1.0)
    mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device)
    
    # 2. 找出需要判断的“未选中面” (Target)
    # 我们只需要计算这些 False 的面，周围是不是 True
    unselected_indices = torch.nonzero(mask_tensor == 0, as_tuple=True)[0]
    num_unselected = len(unselected_indices)
    
    if num_unselected == 0:
        return mask

    print(f"    Target faces to check: {num_unselected}. Processing in batches of {batch_size}...")

    # 准备结果容器
    final_flip_indices = []

    # 3. 分批次进行 KNN 查询 (防止显存爆炸)
    # Source 是所有面 (我们要在所有面里找邻居)
    # Target 是当前批次的未选中面
    
    # knn_points 要求输入维度为 (N, P, D) -> (Batch, Points, Dim)
    # 这里我们将整个点云视为 Batch=1
    source_cloud = all_centers.unsqueeze(0) # (1, Total_Faces, 3)
    
    for i in range(0, num_unselected, batch_size):
        # 获取当前批次的索引
        batch_idx_in_unselected = unselected_indices[i : i + batch_size]
        batch_points = all_centers[batch_idx_in_unselected].unsqueeze(0) # (1, Batch_Size, 3)
        
        # --- A. GPU KNN 查询 ---
        # dists: (1, Batch_Size, K) squared L2 distance
        # idx:   (1, Batch_Size, K) indices of neighbors
        knn_result = knn_points(batch_points, source_cloud, K=k_neighbors)
        dists = knn_result.dists.squeeze(0)  # (Batch_Size, K)
        neighbor_indices = knn_result.idx.squeeze(0) # (Batch_Size, K)
        
        # --- B. 计算高斯权重 ---
        # dists 返回的是平方距离 (squared distance)，所以公式稍微变一下
        # Gaussian = exp( - dist^2 / (2*sigma^2) ) -> 这里 dists 已经是 dist^2 了
        # 加上 1e-6 避免除零
        weights = torch.exp(-dists / (2 * (sigma**2)))
        
        # --- C. 获取邻居的投票状态 ---
        # 使用 gather 从全局 mask 中取出邻居的状态
        # mask_tensor: (Total_Faces) -> neighbor_indices: (Batch, K)
        neighbor_vals = mask_tensor[neighbor_indices] # (Batch_Size, K)
        
        # --- D. 加权平均 ---
        weighted_votes = (neighbor_vals * weights).sum(dim=1)
        total_weights = weights.sum(dim=1) + 1e-8 # 避免除零
        
        scores = weighted_votes / total_weights # (Batch_Size, )
        
        # --- E. 判定 ---
        should_flip_local = scores > threshold
        
        # 记录需要翻转的全局索引
        flip_indices_global = batch_idx_in_unselected[should_flip_local]
        if len(flip_indices_global) > 0:
            final_flip_indices.append(flip_indices_global)
            
        # 清理显存 (可选，但在循环中很有用)
        del dists, neighbor_indices, weights, neighbor_vals, scores
    
    # 4. 应用结果
    new_mask = mask.copy()
    
    if len(final_flip_indices) > 0:
        # 拼接所有批次的结果
        all_indices_to_flip = torch.cat(final_flip_indices).cpu().numpy()
        new_mask[all_indices_to_flip] = True
        print(f"    [GPU] Absorbed {len(all_indices_to_flip)} blind-spot faces.")
    else:
        print("    [GPU] No blind spots filled.")
        
    # 彻底清理显存
    torch.cuda.empty_cache()
    
    return new_mask



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

def absorb_nearby_fragments_gpu(mesh, mask, proximity_threshold=0.02, max_fragment_ratio=0.05, device='cuda'):
    """
    [GPU 加速版] 空间吸附逻辑。
    优化点：
    1. 使用 GPU KNN 一次性计算所有背景面到主体的距离，替代循环查询 KD-Tree。
    2. 使用 CPU 处理连通分量逻辑 (因为这部分非常快且难并行)。
    """
    num_faces = len(mesh.faces)
    max_fragment_size = int(num_faces * max_fragment_ratio)
    
    print(f"  - Running GPU proximity absorption (Dist={proximity_threshold}, MaxRatio={max_fragment_ratio:.1%})...")
    
    # 1. 准备数据
    face_centers = mesh.triangles_center
    face_mask_bool = mask.astype(bool)
    
    if not np.any(face_mask_bool) or np.all(face_mask_bool):
        return mask

    # 将所有面中心转到 GPU
    all_centers_tensor = torch.tensor(face_centers, dtype=torch.float32, device=device)
    
    # 分离“主体”和“背景”
    # Source (主体): 我们要寻找的目标
    # Query (背景): 我们要判断的候选者
    main_indices = torch.nonzero(torch.tensor(face_mask_bool, device=device), as_tuple=True)[0]
    bg_indices = torch.nonzero(torch.tensor(~face_mask_bool, device=device), as_tuple=True)[0]
    
    if len(bg_indices) == 0:
        return mask

    main_points = all_centers_tensor[main_indices]
    bg_points = all_centers_tensor[bg_indices]
    
    # ---------------------------------------------------------
    # 2. GPU 计算：所有背景面到主体的最近距离
    # ---------------------------------------------------------
    # 为了防止显存溢出，如果主体太大，可以降采样主体点云（例如最多只看 50000 个采样点）
    # 但一般 KNN 对 (N, M) 的支持很好，且不需要构建完整矩阵。
    # knn_points inputs: (1, N, 3), (1, M, 3)
    
    # 小技巧：如果 Main Body 巨大 (>100k)，可以随机采样一部分作为“引力场”，
    # 因为我们只需要知道“是不是离得很近”，不需要精确到微米。
    if len(main_points) > 50000:
        perm = torch.randperm(len(main_points), device=device)[:50000]
        search_target = main_points[perm]
    else:
        search_target = main_points

    # 计算距离
    # dists: (1, Num_BG, 1) -> squared L2 distance
    knn_res = knn_points(bg_points.unsqueeze(0), search_target.unsqueeze(0), K=1)
    # 开根号得到真实距离 (Euclidean Distance)
    bg_dists = torch.sqrt(knn_res.dists.squeeze(0).squeeze(1)).cpu().numpy()
    
    # 现在我们有了一个数组 bg_dists，长度等于背景面的数量
    # bg_dists[i] 表示第 i 个背景面距离主体的距离
    
    # ---------------------------------------------------------
    # 3. CPU 拓扑：识别背景中的碎片
    # ---------------------------------------------------------
    inverse_mask = ~face_mask_bool
    adjacency = mesh.face_adjacency
    
    # 仅构建背景面的邻接图
    valid_edges = adjacency[inverse_mask[adjacency].all(axis=1)]
    
    if len(valid_edges) == 0:
        return mask

    rows = valid_edges[:, 0]
    cols = valid_edges[:, 1]
    data = np.ones(len(rows), dtype=bool)
    
    matrix = sp.coo_matrix((data, (rows, cols)), shape=(num_faces, num_faces))
    n_components, labels = sp.csgraph.connected_components(matrix, directed=False)
    
    # 获取所有背景面在“全局”的 label
    # 注意：labels 数组是全长的 (num_faces,)，但在 mask=True 的地方 label 可能是独立的或无效的，我们只关心 mask=False 的地方
    bg_face_global_indices = bg_indices.cpu().numpy()
    bg_labels = labels[bg_face_global_indices]
    
    # ---------------------------------------------------------
    # 4. 聚合决策：碎片是否应该被吸附？
    # ---------------------------------------------------------
    # 我们有一个很棒的数据结构现在：
    # 对于每个背景面，我们知道：1. 它的 Fragment ID (bg_labels), 2. 它离主体的距离 (bg_dists)
    
    # 使用 numpy 快速聚合
    import pandas as pd # 如果环境有 pandas，这是最快的；如果没有，用 numpy 循环
    
    # 为了通用性，我们用 numpy 简单实现聚合
    unique_frag_ids = np.unique(bg_labels)
    
    fragments_absorbed = 0
    faces_absorbed_count = 0
    new_mask = mask.copy()
    
    # 这里虽然有一个循环，但循环次数 = 碎片数量（通常很少，几百个），而不是面数（几十万个）
    # 相比之前的 KD-Tree query，这里只是查简单的数组，极快。
    for frag_id in unique_frag_ids:
        # 找到属于该碎片的背景面索引 (在 bg_indices 中的下标)
        # mask_in_bg_array 是一个 bool 数组，长度为 len(bg_indices)
        mask_in_bg_array = (bg_labels == frag_id)
        
        # 检查大小限制
        count = np.sum(mask_in_bg_array)
        if count > max_fragment_size:
            continue
            
        # 检查距离限制
        # 获取该碎片所有面的距离
        dists_of_fragment = bg_dists[mask_in_bg_array]
        
        # 只要碎片中有一个点足够近，就认为接触了 (Minimum Distance)
        # 或者你也可以用 np.mean(dists_of_fragment) < threshold (平均距离)
        min_dist = np.min(dists_of_fragment)
        
        if min_dist < proximity_threshold:
            # 对应的全局面索引
            # bg_face_global_indices 是所有背景面的全局ID
            # mask_in_bg_array 选出了当前碎片的子集
            faces_to_flip = bg_face_global_indices[mask_in_bg_array]
            new_mask[faces_to_flip] = True
            
            fragments_absorbed += 1
            faces_absorbed_count += count
            
    print(f"    [GPU] Absorbed {fragments_absorbed} fragments ({faces_absorbed_count} faces).")
    return new_mask



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
    [CPU 优选版] 拓扑形态学清理。
    注：连通分量计算是高度串行的图算法，SciPy (C++) 的实现通常比手写 PyTorch 循环更快且更稳定。
    除非拥有专用图计算库 (cuGraph)，否则保持 CPU 处理是最佳选择。
    """
    print(f"  - Running topological cleanup (Hole={min_hole}, Island={min_island})...")
    
    num_faces = len(mesh.faces)
    adjacency = mesh.face_adjacency
    
    # 辅助函数：快速计算连通分量
    def get_components(subset_mask):
        # 1. 筛选出两端都在子集内的边
        # subset_mask[adjacency] 得到 (M, 2) 的 bool 矩阵
        # all(axis=1) 表示这条边的两个面都在 subset 里
        valid_edges = adjacency[subset_mask[adjacency].all(axis=1)]
        
        if len(valid_edges) == 0:
            return np.array([]), 0

        # 2. 构建稀疏矩阵 (只包含有效边)
        # 这是一个无向图
        rows = valid_edges[:, 0]
        cols = valid_edges[:, 1]
        
        # 构建邻接矩阵 (N, N)
        # 注意：这里只用了有效边涉及的节点，但为了索引方便，shape 仍设为 (N, N)
        data = np.ones(len(rows), dtype=bool)
        matrix = sp.coo_matrix(
            (data, (rows, cols)), 
            shape=(num_faces, num_faces)
        )
        
        # 3. 计算连通分量
        n_comps, labels = sp.csgraph.connected_components(matrix, directed=False)
        return labels, n_comps

    refined_mask = mask.copy()
    
    # --- 1. 填补孔洞 (Filling Holes) ---
    inverse_mask = ~refined_mask
    if np.any(inverse_mask):
        labels, n_comps = get_components(inverse_mask)
        if n_comps > 0:
            # 统计每个分量的大小
            # 仅统计 inverse_mask 为 True 的部分的 label
            active_labels = labels[inverse_mask]
            if len(active_labels) > 0:
                unique_labels, counts = np.unique(active_labels, return_counts=True)
                
                # 找出小孔洞的 Label
                small_hole_labels = unique_labels[counts < min_hole]
                
                if len(small_hole_labels) > 0:
                    print(f"    Found {len(small_hole_labels)} small holes to fill.")
                    # 使用 np.isin 快速找到属于这些 label 的所有面
                    # 注意：要与 inverse_mask 取交集，确保只处理背景区域
                    faces_to_fill = np.isin(labels, small_hole_labels) & inverse_mask
                    refined_mask[faces_to_fill] = True

    # --- 2. 移除孤岛 (Removing Islands) ---
    if np.any(refined_mask):
        labels, n_comps = get_components(refined_mask)
        if n_comps > 0:
            active_labels = labels[refined_mask]
            if len(active_labels) > 0:
                unique_labels, counts = np.unique(active_labels, return_counts=True)
                
                small_island_labels = unique_labels[counts < min_island]
                
                if len(small_island_labels) > 0:
                    print(f"    Found {len(small_island_labels)} small islands to remove.")
                    faces_to_remove = np.isin(labels, small_island_labels) & refined_mask
                    refined_mask[faces_to_remove] = False
    
    return refined_mask


def snap_boundary_to_features(mesh, mask, band_width=None, beta=30.0, base_search_ratio=0.05):
    """
    Args:
        base_search_ratio: (float) 基础搜索比例 (对应全选时的最大搜索范围)。
    """
    
    # 1. 预处理 Mask (防报错: 确保是一维布尔数组)
    mask = np.asanyarray(mask).flatten().astype(bool)
    n_total = len(mask)
    n_selected = np.sum(mask)
    
    if n_selected == 0:
        return mask

    # --- [自适应 Bandwidth 计算逻辑] ---
    if band_width is None:
        avg_edge_len = np.mean(mesh.edges_unique_length)
        bbox_diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
        
        # 计算尺度因子 (sqrt of area fraction)
        area_fraction = n_selected / n_total
        scale_factor = np.sqrt(area_fraction)
        scale_factor = np.clip(scale_factor, 0.1, 1.0)
        
        # 计算目标半径
        current_ratio = base_search_ratio * scale_factor
        target_radius = bbox_diag * current_ratio
        
        # 转换为跳数
        adaptive_hops = int(target_radius / avg_edge_len)
        band_width = np.clip(adaptive_hops, 2, 30)
        
        print(f"  - [Adaptive Info]")
        print(f"    Selected: {n_selected}/{n_total} faces ({area_fraction*100:.1f}%)")
        print(f"    Scale Factor: {scale_factor:.3f}")
        print(f"    Target Radius: {target_radius:.4f} (Ratio: {current_ratio:.4f})")
        print(f"    Calculated Bandwidth: {band_width} hops")
    else:
        print(f"  - [Fixed] Using fixed Bandwidth: {band_width} hops")

    print(f"  - Running Graph Cut snap (Band={band_width}, Beta={beta})...")
    
    adjacency = mesh.face_adjacency
    face_normals = mesh.face_normals
    
    adj_graph = nx.Graph()
    adj_graph.add_edges_from(adjacency)
    
    # 找出边界
    is_boundary_edge = mask[adjacency[:, 0]] ^ mask[adjacency[:, 1]]
    boundary_faces_indices = np.unique(adjacency[is_boundary_edge])
    
    if len(boundary_faces_indices) == 0:
        return mask

    try:
        # [CRITICAL FIX HERE]
        # 必须将 numpy array 转为 set 或 list，否则 NetworkX 内部做 if sources: 判断时会崩溃
        sources_set = set(boundary_faces_indices.tolist())
        
        dists = nx.multi_source_dijkstra_path_length(adj_graph, sources_set, cutoff=band_width)
    except Exception as e:
        # 打印详细错误堆栈以便后续排查，虽然现在应该修好了
        import traceback
        traceback.print_exc()
        print(f"    Dijkstra failed: {e}. Skipping snap.")
        return mask
        
    roi_nodes = list(dists.keys())
    
    if not roi_nodes:
        return mask
        
    G = nx.Graph()
    SOURCE = 'source'
    SINK = 'sink'
    
    for node in roi_nodes:
        dist = dists[node]
        
        # 如果超出带宽，强制根据当前 mask 归类
        if dist >= band_width:
            # 显式转换 bool，防止 numpy bool 歧义
            if bool(mask[node]): 
                G.add_edge(SOURCE, node, capacity=float('inf'))
            else:
                G.add_edge(node, SINK, capacity=float('inf'))
    
    roi_set = set(roi_nodes)
    processed_edges = set()
    
    # 构建图割权重
    for u in roi_nodes:
        if u not in adj_graph: continue
        for v in adj_graph[u]:
            # 确保只在 ROI 内部建立边
            if v in roi_set and v > u:
                edge_key = (u, v)
                if edge_key in processed_edges: continue
                processed_edges.add(edge_key)
                
                n1 = face_normals[u]
                n2 = face_normals[v]
                
                dot = np.clip(np.dot(n1, n2), 0.0, 1.0)
                # Beta 越高，越倾向于在棱角处切割
                capacity = 1.0 + beta * np.exp(5.0 * (dot - 1.0)) 
                G.add_edge(u, v, capacity=capacity)

    try:
        cut_value, partition = nx.minimum_cut(G, SOURCE, SINK)
        reachable, non_reachable = partition
        
        new_mask = mask.copy()
        count = 0
        
        for node in roi_nodes:
            new_val = (node in reachable)
            if new_mask[node] != new_val:
                new_mask[node] = new_val
                count += 1
                
        print(f"    Snapped boundary. Flipped {count} faces.")
        return new_mask
        
    except Exception as e:
        print(f"    Graph cut failed: {e}. Returning original mask.")
        return mask

def smooth_boundary_geometry(mesh, mask, iterations=10, strength=0.5):
    print(f"  - Running geometric boundary smoothing ({iterations} iters)...")
    
    # [修复 1] 强制展平并转为纯布尔型
    mask = np.asanyarray(mask).flatten().astype(bool)
    
    adjacency = mesh.face_adjacency
    adjacency_edges = mesh.face_adjacency_edges
    
    # [修复 2] 确保 is_boundary_edge 是严格的 (M,) 一维数组
    is_boundary_edge = mask[adjacency[:, 0]] ^ mask[adjacency[:, 1]]
    is_boundary_edge = is_boundary_edge.flatten()
    
    # 这样索引 adjacency_edges (M, 2) 就不会有歧义
    boundary_edges = adjacency_edges[is_boundary_edge]
    
    if len(boundary_edges) == 0:
        return
        
    g = nx.Graph()
    g.add_edges_from(boundary_edges)
    boundary_nodes = list(g.nodes())
    
    if not boundary_nodes:
        return

    # 获取当前顶点 (引用)
    current_vertices = mesh.vertices 
    
    for _ in range(iterations):
        new_positions = {}
        for v_idx in boundary_nodes:
            neighbors = list(g.neighbors(v_idx))
            if not neighbors: continue
            
            # 简单的拉普拉斯平滑
            # 注意：neighbors 是索引列表，可以直接用于 numpy 索引
            neighbor_verts = current_vertices[neighbors]
            centroid = np.mean(neighbor_verts, axis=0)
            
            current_pos = current_vertices[v_idx]
            new_positions[v_idx] = current_pos + (centroid - current_pos) * strength
            
        # 批量更新，防止迭代中的位置依赖混乱
        if new_positions:
            indices = list(new_positions.keys())
            values = np.array(list(new_positions.values()))
            current_vertices[indices] = values
            
    mesh.vertices = current_vertices
    # 重新计算法线以匹配新的几何形状
    mesh.fix_normals()
    print(f"    Smoothed {len(boundary_nodes)} boundary vertices.")



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
    positive_bias_weight = 2
    
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
    
    # 1. 初始阈值化 (Initial Thresholding)
    is_selected_tensor = refined_ratios > vote_threshold
    mask_bool = is_selected_tensor.cpu().numpy() 

    # 2. [GPU] 空间高斯填充 (解决花蕊/盲区问题)
    # 先把该选的都选上，哪怕选多了一点也没关系
    mask_bool = absorb_blind_spots_gpu(
        mesh_normalized, 
        mask_bool, 
        sigma=0.08,        
        k_neighbors=40,   
        threshold=0.6,     # 周围加权能量 > 0.6 才认为是内部
        batch_size=20000,  # 显存如果 < 8GB，建议 10000；> 12GB 可以 50000
        device=device      # 传入你的 device 变量 (e.g. "cuda:0")
    )

    
    # 3. [GPU] 碎片吸附 (解决周围飘散的小碎块)
    mask_bool = absorb_nearby_fragments_gpu(
        mesh_normalized,
        mask_bool,
        proximity_threshold=0.04,  # If the distance is less than 0.04 (unit length), it's considered part of the same object
        max_fragment_ratio=0.08     # Only absorb fragments that are less than 8% of the total faces
    )
    
    # 4. [CPU] 拓扑清理 (填洞 & 去孤岛)
    # 保证 Mask 在拓扑上是干净的，没有单像素噪音
    mask_bool = apply_morphological_cleanup(
        mesh_normalized, 
        mask_bool, 
        min_island=50,      
        min_hole=50 
    )
    
    
    # 5. 特征对齐 (Feature Snapping)
    # 这一步会移动 Mask 边界，让它自动“卡”进法线变化的沟槽里。
    mask_bool = snap_boundary_to_features(
        mesh_normalized,
        mask_bool,
        band_width=None,    # 开启自适应
        base_search_ratio=0.4,   # 搜索范围约为物体尺寸的 2%
        beta=50,
    )

    # 6. [CPU] 几何平滑 (Geometric Smoothing)
    # 最后一步，把剩下的一点点锯齿拉直，让边缘圆润
    # 注意：这会修改 Mesh 顶点，放在最后做
    smooth_boundary_geometry(
        mesh_normalized,
        mask_bool,
        iterations=10,
        strength=0.5
    )


    selected_face_ids = np.where(mask_bool)[0]

    print(f"Splitting mesh... Selected {len(selected_face_ids)} faces out of {num_faces}.")

    all_indices = np.arange(num_faces)
    unselected_face_ids = np.setdiff1d(all_indices, selected_face_ids)
    
    scene_parts = []
    
    if len(selected_face_ids) > 0:
        part_target = mesh_normalized.submesh([selected_face_ids], append=True)
        part_target.metadata['name'] = 'part_target'
        part_target.visual.vertex_colors = get_random_pastel_color()
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
import open3d as o3d
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
from matplotlib.patches import Ellipse
from typing import Literal
from utils.pytorch3d_compat import quaternion_to_matrix, matrix_to_quaternion
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import pyvista as pv
import os

def save_labeled_points(points, label,save_path):
    """
    Save labeled points to a file.
    Args:
        points: (N, 3) numpy array of point coordinates
        label: (N,) numpy array of labels
    """
    assert points.shape[0] == label.shape[0], "Points and labels must have the same number of elements."
    colors = np.random.rand(len(np.unique(label)), 3)  # Random colors for each label
    point_colors = np.array([colors[l] if l != -1 else [0.8,0.8,0.8] for l in label])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    o3d.io.write_point_cloud(os.path.join(save_path,"labeled_points.ply"), pcd)

def z_axis_to_vector_rotation_torch(target_vector: torch.Tensor, target: str = 'cylinder') -> torch.Tensor:
    """
    Compute rotation matrix that aligns [0,0,1] (or [1,0,0]) to target_vector.

    Args:
        target_vector: Tensor of shape (3,) — the vector to align to
        target: 'cylinder' (default, align from z-axis) or 'gs' (align from x-axis)

    Returns:
        rot_matrix: Tensor of shape (3, 3)
    """
    target_vector = target_vector / target_vector.norm(p=2, dim=0, keepdim=False)

    if target == 'cylinder':
        z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=target_vector.dtype, device=target_vector.device)
    elif target == 'gs':
        z_axis = torch.tensor([1.0, 0.0, 0.0], dtype=target_vector.dtype, device=target_vector.device)
    else:
        raise ValueError("target must be 'gs' or 'cylinder'")
    z_axis =z_axis.unsqueeze(0).repeat(target_vector.shape[0], 1)
    v = torch.cross(z_axis, target_vector)
    c = torch.sum(z_axis *target_vector)

    if torch.isclose(c, torch.tensor(1.0, device=target_vector.device)):
        return torch.eye(3, dtype=target_vector.dtype, device=target_vector.device)
    if torch.isclose(c, torch.tensor(-1.0, device=target_vector.device)):
        # Return 180 degree rotation around an orthogonal axis
        return torch.eye(3, dtype=target_vector.dtype, device=target_vector.device) * torch.tensor([-1, -1, 1], device=target_vector.device).unsqueeze(0)

    # Skew-symmetric cross-product matrix
    vx = torch.zeros((target_vector.shape[0], 3, 3), dtype=target_vector.dtype, device=target_vector.device)
    vx[:, 0, 1] = -v[:, 2]
    vx[:, 0, 2] = v[:, 1]
    vx[:, 1, 0] = v[:, 2]
    vx[:, 1, 2] = -v[:, 0]
    vx[:, 2, 0] = -v[:, 1]
    vx[:, 2, 1] = v[:, 0]

    rot_matrix = torch.eye(3, dtype=target_vector.dtype, device=target_vector.device) + vx + vx @ vx * (1 / (1 + c))
    return rot_matrix

def align_z_axis_to_vector(target_vec: torch.Tensor) -> torch.Tensor:
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=target_vec.device)
    v = torch.cross(z_axis, target_vec)
    c = torch.dot(z_axis, target_vec)
    s = torch.norm(v)

    if s < 1e-6:
        return torch.eye(3, device=target_vec.device)

    vx = torch.tensor([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], device=target_vec.device)

    R = torch.eye(3, device=target_vec.device) + vx + vx @ vx * ((1 - c) / (s**2 + 1e-8))
    return R

def estimate_gs_para_from_cluster(xyz: torch.Tensor, anisotropy_threshold: float=3.0, device='cuda'):
    """
    Estimate Gaussian parameters (center, rotation, scale) from a point cluster using PCA.
    Input:
        xyz: (N, 3) torch tensor
    Returns:
        center: (3,) mean of points
        rot_gs: (4,) quaternion [w, x, y, z]
        scale: (3,) sqrt of eigenvalues (clipped)
    """
    xyz = xyz.to(device)
    center = xyz.mean(dim=0)  # (3,)
    cov = torch.cov(xyz.T)  # (3, 3)

    # Eigen decomposition
    eigvals, eigvecs = torch.linalg.eigh(cov)
    idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    major_axis = eigvecs[:, 0]
    normal = eigvecs[:, 2]

    # Rotation matrices
    rot_matrix_cylinder = align_z_axis_to_vector(major_axis)
    rot_matrix_disk = align_z_axis_to_vector(normal)
    rot_matrix = eigvecs # local frame from PCA

    # Convert matrix to quaternion [w, x, y, z]
    rot_gs = matrix_to_quaternion(rot_matrix[None])[0]  # (4,)
    scale = torch.sqrt(eigvals)
    
    
    # leaf/branch prior
    anisotropy = scale[0] / scale[1]
    if anisotropy < anisotropy_threshold:
        prior = torch.tensor(0.2, device=xyz.device)  # leaf prior
    else:
        prior = torch.tensor(0.8, device=xyz.device)  # branch prior
    return center, rot_gs, scale,  anisotropy,rot_matrix_cylinder,rot_matrix_disk
       

def strpr_to_cylinder(pos, S,  R, iteration, create_mesh=False, save_path = 'output',resolution=32):
    """
    Convert structural primitives (StrPrs) to cylinder surfaces for branch modeling.
    issue with rotation rebuild, a right version is directly using the rotation_cylinder -> main_axis = rot_matrix[:,:,2] -> R_align = align_z_axis_to_vector(main_axis[i]) ->mesh rotate(_torch_to_numpy(R_align), center=(0, 0, 0))

    Args:
        p: (N, 3) positions
        S: (N, 2) scaling — assuming S[:,0]=height, S[:,1]=radius for 2DGS
        R: (N, 3,3) rotation matrix
    Returns:
        dict of geometric params, mesh_all, mesh_list
    """

    rot_matrix = R# Convert quaternion to rotation matrix

    r = S[:, 2]      # radius = smallest scale (the true branch thickness; ×3 over-fattened
                     # the tube so it captured leaves' planar spread -> branch-bias in binding)
    h = 3 * S[:, 0]  # height
    main_axis = rot_matrix[:,:,0] # z-axis of the cylinder 2 for rot_cylinder
    cylinder_params ={
        'pos': pos,
        'radius': r,
        'height': h,
        'axis_vec': main_axis,
    }
    if create_mesh:
        # Save the cylinder parameters
        mesh_all = o3d.geometry.TriangleMesh()
        mesh_list = []
        for i in range(len(pos)):
            mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=r[i], height=h[i], resolution=resolution, split=20)
            R_align = align_z_axis_to_vector(main_axis[i])  # (3, 3)
            mesh.rotate(_torch_to_numpy(R_align), center=(0, 0, 0))
            mesh.translate(_torch_to_numpy(pos[i]))
            mesh_list.append(mesh)
            mesh_all += mesh
        o3d.io.write_triangle_mesh(save_path, mesh_all)
        return cylinder_params, mesh_all, mesh_list
    else:
        return cylinder_params, None, None

def strpr_to_disk(pos, S, R, iteration,resolution=50, create_mesh=False,save_path='output'):
    rot_matrix = R
    u = rot_matrix[:, :, 0]  # disk x-axis (major)
    v = rot_matrix[:, :, 1]  # disk y-axis (minor)
    n = rot_matrix[:, :, 2]  # disk normal
    a = 2 * S[:, 0]  # major axis radius
    b =     S[:, 1]  # minor axis radius
    disk_params = {'center': pos, 'a': a, 'b': b, 'u': u, 'v': v, 'n': n}
    if not create_mesh:
        return disk_params, None, None

    mesh_all = o3d.geometry.TriangleMesh()
    mesh_all_list = []

    theta = torch.linspace(0, 2 * torch.pi, steps=resolution + 1, device=pos.device)[:-1]
    cos_t = torch.cos(theta)  # (res,)
    sin_t = torch.sin(theta)  # (res,)

    for i in range(pos.shape[0]):
        x = a[i] * cos_t  # (res,)
        y = b[i] * sin_t  # (res,)
        verts_ring = pos[i] + x.unsqueeze(1) * u[i] + y.unsqueeze(1) * v[i]  # (res, 3)
        verts = torch.cat([pos[i].unsqueeze(0), verts_ring], dim=0)  # (res+1, 3)

        # Fan triangulation
        tri_idx = torch.stack([
            torch.zeros(resolution, dtype=torch.long, device=pos.device),
            (torch.arange(1, resolution + 1, device=pos.device) % resolution) + 1,
            torch.arange(1, resolution + 1, device=pos.device)
        ], dim=1)

        # Build Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts.detach().cpu().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(tri_idx.detach().cpu().numpy())
        mesh_all_list.append(mesh)
        mesh_all += mesh
    o3d.io.write_triangle_mesh(save_path, mesh_all)
    return disk_params, mesh_all, mesh_all_list

def gs_to_cylinder_distance(xyz, nn, cylinder_params):
    """
    Compute the distance from points to cylinder surfaces.
    Args:
        xyz: (N, 3) torch tensor of point coordinates
        nn: (N, 3) correspondence between points and cylinder 
        cylinder_params: dict with keys 'pos', 'radius', 'height', 'axis_vec'
    Returns:
        dist: (N,) torch tensor of distances to cylinder surfaces
    """
    C = cylinder_params['pos'][nn]    # (M,3)
    u = cylinder_params['axis_vec'][nn]      # (M,3) normalized
    r = cylinder_params['radius'][nn]               # (M,)
    h = cylinder_params['height'][nn]             # (M,)
    xyz = xyz.to(C.device)  # Ensure xyz is on the same device as C
    v = xyz - C
    t = torch.sum(v * u, dim=-1)                  # projection
    t_c = torch.clamp(t, -h, h)
    P_a = C + t_c.unsqueeze(1) * u                # nearest on axis

    d_axis = torch.norm(xyz - P_a, dim=-1)        # radial distance
    d_side = torch.abs(d_axis - r)                # penalize inside & outside

    cap_mask = (t.abs() > h)
    if cap_mask.any():
        d_cap = torch.sqrt(
            (d_side[cap_mask])**2 + (t[cap_mask].abs() - h[cap_mask])**2
        )
        d_side[cap_mask] = d_cap
    return d_side

def gs_to_disk_distance(xyz, parent, disk_param):
    C = disk_param['center'][parent]
    n = disk_param['n'][parent]
    a = disk_param['a'][parent]
    b = disk_param['b'][parent]
    xyz = xyz.to(C.device)  # Ensure xyz is on the same device as C
    v = xyz - C
    d_plane = torch.abs(torch.sum(v * n, dim=-1))

    tmp = torch.tensor([1.,0.,0.], device=xyz.device).expand_as(n)
    e1 = F.normalize(torch.cross(n, tmp), dim=-1)
    bad = torch.isnan(e1).any(dim=-1)
    if bad.any():
        tmp2 = torch.tensor([0.,1.,0.], device=xyz.device).expand_as(n[bad])
        e1[bad] = F.normalize(torch.cross(n[bad], tmp2), dim=-1)
    e2 = torch.cross(n, e1)

    x_coord = torch.sum(v * e1, dim=-1)
    y_coord = torch.sum(v * e2, dim=-1)

    eps = 1e-6
    r_val = torch.sqrt((x_coord / (a + eps))**2 + (y_coord / (b + eps))**2)

    x_ellipse = (x_coord / r_val.clamp(min=eps)) * a
    y_ellipse = (y_coord / r_val.clamp(min=eps)) * b
    d_edge = torch.sqrt((x_coord - x_ellipse)**2 + (y_coord - y_ellipse)**2)

    d = torch.where(r_val <= 1.0, d_plane, torch.sqrt(d_plane**2 + d_edge**2))
    return d


def _torch_to_numpy(x: torch.Tensor):
    return x.detach().cpu().numpy()


# def build_mst_from_endpoints(top, bottom, k:int=3):
#     top = top.detach().cpu().numpy()
#     bottom = bottom.detach().cpu().numpy()
#     N = top.shape[0]
#     points = np.empty((2*N, 3), dtype=np.float32)
#     points[0::2] = top
#     points[1::2] = bottom    
#     M = points.shape[0]
#     row, col, data = [], [], []

#     # (1)  zero‑weight internal edges
#     idx = np.arange(N, dtype=np.int32)
#     row.extend(2*idx)           ; col.extend(2*idx+1)
#     data.extend(np.zeros(N))    # weight 0
#     #  symmetric entry
#     row.extend(2*idx+1)         ; col.extend(2*idx)
#     data.extend(np.zeros(N))

#     # (2)  k‑NN edges  (Euclidean distance)
#     tree = cKDTree(points)
#     dists, neigh = tree.query(points, k=k+1)     # first neighbour is itself

#     for i in range(M):
#         for j, d in zip(neigh[i, 1:], dists[i, 1:]):   # skip self
#             row.append(i);  col.append(j);  data.append(d)
#             # symmetric entry
#             row.append(j);  col.append(i);  data.append(d)

#     # ---------- build symmetric CSR adjacency ----------
#     A = csr_matrix((data, (row, col)), shape=(M, M))

#     # ---------- Minimum Spanning Tree ----------
#     T_csr  = minimum_spanning_tree(A)            # still CSR  (M×M)
#     mst_edges = np.vstack(T_csr.nonzero()).T     # (M-1, 2)  index pairs
#     mst_w     = T_csr.data  
#     return mst_edges, points

def save_mst_ply(points, edges, path='mst.ply'):
    from plyfile import PlyElement, PlyData
    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
    points = np.asarray(points, dtype=np.float32)
    if torch.is_tensor(edges):
        edges = edges.detach().cpu().numpy()
    edges = np.asarray(edges)

    v = np.empty(points.shape[0], dtype=[('x','f4'),('y','f4'),('z','f4')])
    v['x'], v['y'], v['z'] = points.T
    e = np.empty(edges.shape[0], dtype=[('vertex1','u4'),('vertex2','u4')])
    e['vertex1'] = edges[:,0];  e['vertex2'] = edges[:,1]
    PlyData([PlyElement.describe(v,'vertex'),
             PlyElement.describe(e,'edge')], text=True).write(path)

# def build_mst_from_endpoints(endpoints, strpr_pairs, k=8):
#     import networkx as nx
#     M = endpoints.shape[0]
#     endpoints_np = endpoints.detach().cpu().numpy()

#     # kNN
#     from sklearn.neighbors import NearestNeighbors
#     nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(endpoints_np)
#     distances, indices = nbrs.kneighbors(endpoints_np)

#     edges = []
#     for i in range(M):
#         for j, d in zip(indices[i,1:], distances[i,1:]): 
#             edges.append((i,j,float(d)))

#     # inner edges
#     if strpr_pairs is not None:
#         for a,b in strpr_pairs.tolist():
#             edges.append((a,b,1e-6))

#     # cross edges
#     G = nx.Graph()
#     G.add_weighted_edges_from(edges)
#     T = nx.minimum_spanning_tree(G, weight='weight')

#     tree_edges = np.array(list(T.edges()))
#     return tree_edges

def build_mst_from_endpoints(endpoints, strpr_pairs, k=16):
    """
    endpoints: (M,3) torch.Tensor
    strpr_pairs: (S,2) torch.LongTensor or None
    return: tree_edges [E,2] (numpy)
    加入：轴向/切向连续性惩罚（通过局部 PCA 估计每点主轴）
    """
    import numpy as np
    import networkx as nx
    from sklearn.neighbors import NearestNeighbors
    from numpy.linalg import svd

    M = endpoints.shape[0]
    X = endpoints.detach().cpu().numpy()

    # ---------- 近邻图（候选 & 局部统计） ----------
    k_eff = min(k+1, M)
    knn = NearestNeighbors(n_neighbors=k_eff, algorithm='auto').fit(X)
    dists, idxs = knn.kneighbors(X)

    # 局部尺度/长度统计
    local_scale = np.median(dists[:, 2:], axis=1) if dists.shape[1] > 2 else np.median(dists[:, 1:], axis=1)
    mu_local  = np.mean(dists[:, 1:], axis=1)
    std_local = np.std(dists[:, 1:], axis=1) + 1e-8

    # ---------- 局部 PCA 估计每个端点主轴 u_i ----------
    # 用稍大一点的邻域更稳
    k_dir = min(max(12, k_eff), M)
    knn_dir = NearestNeighbors(n_neighbors=k_dir, algorithm='auto').fit(X)
    _, idxs_dir = knn_dir.kneighbors(X)

    U = np.zeros_like(X)  # 每点主轴
    for i in range(M):
        P = X[idxs_dir[i]]  # (k_dir, 3)
        C = P - P.mean(axis=0, keepdims=True)
        # SVD: 第一主成分对应最大奇异值方向
        try:
            _, _, Vt = svd(C, full_matrices=False)
            ui = Vt[0]  # (3,)
        except np.linalg.LinAlgError:
            ui = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        nrm = np.linalg.norm(ui) + 1e-12
        U[i] = ui / nrm

    # ---------- 占据计数（穿空惩罚） ----------
    rad = NearestNeighbors(radius=1.0, algorithm='auto').fit(X)

    # ---------- 超参（可调） ----------
    alpha_len = 1.0        # 长度 z-score 权重
    beta_void = 1.5        # 穿空惩罚权重
    gamma_axis = 1.0       # 轴向一致性惩罚权重
    gamma_tan  = 1.0       # 切向一致性惩罚权重

    samples_min = 20
    rho_scale = 1
    count_thresh_factor = 4
    max_samples = 32

    def edge_cost(i, j, dij):
        # --- 长度异常 z-score ---
        mu_ij  = 0.5 * (mu_local[i] + mu_local[j])
        std_ij = 0.5 * (std_local[i] + std_local[j])
        z = abs(dij - mu_ij) / std_ij

        # --- 穿空比例 p_void ---
        step = 0.5 * float(max(1e-6, min(local_scale[i], local_scale[j])))
        K = int(max(samples_min, min(max_samples, np.ceil(dij / step))))
        tlin = np.linspace(0.0, 1.0, K, dtype=np.float32)[:, None]
        S = X[i][None,:]*(1.0-tlin) + X[j][None,:]*tlin
        rho = float(rho_scale * step)
        rad.set_params(radius=rho)
        neighs = rad.radius_neighbors(S, return_distance=False)
        exp_cnt = max(4.0, 8.0 * (rho / (1e-6 + 0.5*(local_scale[i]+local_scale[j])))**3)
        thr = count_thresh_factor * exp_cnt
        p_void = float(np.mean([len(ii) < thr for ii in neighs]))

        # --- 轴向/切向一致性（夹角惩罚） ---
        tvec = (X[j] - X[i]) / (dij + 1e-12)
        ui, uj = U[i], U[j]

        cos_ij = abs(np.dot(ui, uj))  # 两端主轴一致性（无向）
        cos_ti = abs(np.dot(tvec, ui))
        cos_tj = abs(np.dot(tvec, uj))
        pen_axis = 1.0 + gamma_axis * (1.0 - cos_ij)
        pen_tan  = 1.0 + gamma_tan  * (1.0 - 0.5*(cos_ti + cos_tj))

        # --- 综合 ---
        penalty_geom = (1.0 + alpha_len * z) * (1.0 + beta_void * p_void)
        w = float(dij) * penalty_geom * pen_axis * pen_tan
        return w

    # ---------- 候选边 ----------
    edges = []
    for i in range(M):
        for j, d in zip(idxs[i,1:], dists[i,1:]):
            if j <= i:
                continue
            w = edge_cost(i, int(j), float(d))
            edges.append((i, int(j), w))

    # 内部强制边（极小权重）
    if strpr_pairs is not None and len(strpr_pairs) > 0:
        pairs = strpr_pairs.detach().cpu().numpy()
        for a, b in pairs.tolist():
            if a == b: 
                continue
            ai, bj = int(min(a,b)), int(max(a,b))
            edges.append((ai, bj, 1e-6))

    # ---------- 构图 ----------
    G = nx.Graph()
    G.add_weighted_edges_from(edges)

    # ---------- 跨分量补桥（确保单连通） ----------
    if not nx.is_connected(G):
        k2 = min(max(32, k_eff*2), M)
        knn2 = NearestNeighbors(n_neighbors=k2, algorithm='auto').fit(X)
        d2, i2 = knn2.kneighbors(X)

        def connect_components(G):
            comps = [list(c) for c in nx.connected_components(G)]
            if len(comps) == 1:
                return G, False
            comp_id = np.full(M, -1, dtype=int)
            for cid, nodes in enumerate(comps):
                comp_id[nodes] = cid
            best = None
            for u in range(M):
                cu = comp_id[u]
                for v, d in zip(i2[u,1:], d2[u,1:]):
                    v = int(v)
                    if comp_id[v] == cu:
                        continue
                    w = edge_cost(u, v, float(d))
                    if (best is None) or (w < best[0]):
                        best = (w, u, v)
            if best is not None:
                w,u,v = best
                G.add_edge(u, v, weight=w)
                return G, True
            return G, False

        changed = True
        while changed and (not nx.is_connected(G)):
            G, changed = connect_components(G)

    # ---------- MST ----------
    T = nx.minimum_spanning_tree(G, weight='weight')
    import numpy as np
    tree_edges = np.array(list(T.edges()))
    return tree_edges


def compute_node_radius_from_edge_radius(
    num_nodes: int,
    edges: np.ndarray,          # (E,2) int
    edge_radii: np.ndarray,     # (E,) float
    reduce: Literal["max", "mean"] = "max",
    fallback: float = 0.0,
) -> np.ndarray:
    edges = np.asarray(edges, dtype=np.int64)
    edge_radii = np.asarray(edge_radii.detach().cpu(), dtype=np.float32)

    node_r = np.full((num_nodes,), np.nan, dtype=np.float32)
    # 累积
    if reduce == "max":
        node_r[:] = -np.inf
        for (u, v), r in zip(edges, edge_radii):
            node_r[u] = max(node_r[u], r)
            node_r[v] = max(node_r[v], r)
        node_r[~np.isfinite(node_r)] = fallback
    else:
        # mean
        acc = np.zeros((num_nodes,), dtype=np.float64)
        cnt = np.zeros((num_nodes,), dtype=np.int64)
        for (u, v), r in zip(edges, edge_radii):
            acc[u] += float(r); cnt[u] += 1
            acc[v] += float(r); cnt[v] += 1
        node_r = np.where(cnt > 0, (acc / np.maximum(1, cnt)), fallback).astype(np.float32)
    # 最小半径下限，避免 tube 崩塌
    eps = 1e-6
    node_r = np.maximum(node_r, eps)
    return node_r


def build_polyline_graph(points: np.ndarray, edges: np.ndarray, node_radius: np.ndarray) -> pv.PolyData:
    if hasattr(points, "detach"):
        points = points.detach().cpu().numpy()
    points = np.asarray(points, dtype=np.float32)
    edges = np.asarray(edges, dtype=np.int64)
    node_radius = np.asarray(node_radius, dtype=np.float32)

    lines = np.empty(edges.shape[0] * 3, dtype=np.int64)
    lines[0::3] = 2
    lines[1::3] = edges[:, 0]
    lines[2::3] = edges[:, 1]

    poly = pv.PolyData(points)
    poly.lines = lines
    poly["radius"] = node_radius  # point_data 标量
    return poly

def prune_leaf_strpr_by_radius_growth(tree_edges_ep, node_r_ep, tau=0.30, root=None, max_passes=10):
    """
    Radius-growth sanity check at the StrPr level, based on an endpoint-level MST.

    Args:
        tree_edges_ep : (E,2) ndarray of int
            Endpoint-level MST edges over 2*N_strpr nodes (0..2N-1).
        node_r_ep     : (2*N,) ndarray of float
            Radius per endpoint node.
        tau           : float
            Prune leaf StrPr if r_child > (1+tau)*r_parent.
        root          : int or None
            Root StrPr id. If None, use the max-radius StrPr as root.
        max_passes    : int
            Max pruning iterations.

    Returns:
        prune_mask_strpr : (N,) bool
            True for StrPr to prune.
        kept_edges_ep    : (E_kept,2) ndarray
            Endpoint-level edges after removing pruned StrPr endpoints.
        parent_strpr     : (N,) int
            Parent StrPr id for each node (-1 for root or pruned).
    """
    tree_edges_ep = np.asarray(tree_edges_ep, dtype=np.int64)
    node_r_ep     = np.asarray(node_r_ep, dtype=np.float32)
    assert tree_edges_ep.ndim == 2 and tree_edges_ep.shape[1] == 2, "tree_edges_ep must be (E,2)"
    M = node_r_ep.shape[0]
    assert M % 2 == 0, "Endpoint count must be even (2*N_strpr)."
    N = M // 2  # number of StrPr

    # --- StrPr 半径：取两个端点半径的均值（可按需换成 max/median）
    r_strpr = (node_r_ep[0::2] + node_r_ep[1::2]) * 0.5  # (N,)

    # --- 端点对（内部边）：(2i,2i+1)
    def is_inner_edge(u, v):
        return (u // 2) == (v // 2)

    # --- 压缩为 StrPr 级边（只保留跨 StrPr；多条端点跨边归并为一条）
    edges_sp = set()
    for u, v in tree_edges_ep:
        su, sv = u // 2, v // 2
        if su != sv:
            a, b = (su, sv) if su < sv else (sv, su)
            edges_sp.add((a, b))
    edges_sp = np.array(list(edges_sp), dtype=np.int64)  # (E_sp,2)

    # ---- 若端点 MST 极端稀疏导致 StrPr 图不连通，照样逐分量处理 ----
    # 先构图
    adj_sp = defaultdict(list)
    for a, b in edges_sp:
        adj_sp[a].append(b); adj_sp[b].append(a)

    # 选根：最大半径 StrPr（或用户给定）
    if root is None:
        root = int(np.argmax(r_strpr))

    # BFS 得到 parent/depth（在 StrPr 图里）
    parent = -np.ones(N, dtype=np.int64)
    depth  = -np.ones(N, dtype=np.int64)
    q = deque([root])
    parent[root] = -1
    depth[root]  = 0
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj_sp[u]:
            if depth[v] == -1:
                parent[v] = u
                depth[v]  = depth[u] + 1
                q.append(v)

    # 度数（StrPr 级，仅跨边）
    deg = np.zeros(N, dtype=np.int32)
    for a, b in edges_sp:
        deg[a] += 1
        deg[b] += 1

    prune_mask = np.zeros(N, dtype=bool)
    thr = 1.0 + float(tau)

    # 迭代：只剪叶子（deg==1）且半径增长异常的 StrPr
    for _ in range(max_passes):
        to_prune = []
        for v in range(N):
            if prune_mask[v] or v == root:
                continue
            if deg[v] == 1:  # leaf StrPr only
                p = parent[v]
                if (p >= 0) and (not prune_mask[p]):
                    if r_strpr[v] > thr * r_strpr[p]:
                        to_prune.append(v)
        if not to_prune:
            break
        # 应用剪枝：标记并更新度
        for v in to_prune:
            prune_mask[v] = True
            for u in adj_sp[v]:
                if not prune_mask[u]:
                    deg[u] = max(0, deg[u] - 1)
            deg[v] = 0

    # --- 把被剪的 StrPr 映射回端点，过滤端点级边 ---
    prune_mask_ep = np.repeat(prune_mask, 2)  # [2*N], True for both endpoints of pruned StrPr
    kept_edges_ep = []
    for u, v in tree_edges_ep:
        if (not prune_mask_ep[u]) and (not prune_mask_ep[v]):
            kept_edges_ep.append((u, v))
    kept_edges_ep = np.asarray(kept_edges_ep, dtype=np.int64)

    # 被剪节点 parent 置 -1
    parent[prune_mask] = -1

    return prune_mask, kept_edges_ep, parent
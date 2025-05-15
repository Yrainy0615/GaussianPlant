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
from pytorch3d.transforms import quaternion_to_matrix
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

def estimate_laplacian(pcd, radius=0.1):
    # 计算点云的邻域
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    points = np.asarray(pcd.points)
    n_points = len(points)
    
    # 构建拉普拉斯矩阵 L
    rows = []
    cols = []
    data = []
    
    for i in range(n_points):
        [k, idx, _] = pcd_tree.search_radius_vector_3d(points[i], radius)
        k_neighbors = len(idx)  # 包括自身
        for j in range(k_neighbors):
            rows.append(i)
            cols.append(idx[j])
            data.append(-1.0)
        rows.append(i)
        cols.append(i)
        data.append(k_neighbors - 1)
    
    L = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(n_points, n_points))
    return L

def laplacian_contraction(pcd, num_iterations=20, radius=0.1, lambda_L=0.5, lambda_H=1.0):
    points = np.asarray(pcd.points)
    n_points = len(points)

    # 初始化收缩权重
    W_L = np.eye(n_points) * lambda_L
    W_H = np.eye(n_points) * lambda_H
    
    # 迭代收缩过程
    for t in range(num_iterations):
        L = estimate_laplacian(pcd, radius)
        
        # 计算 P^{t+1}
        rhs = np.zeros((n_points, 3))
        rhs[:, :] = np.dot(W_H, points)
        
        lhs = np.vstack([np.zeros((1, 3)), np.dot(W_L, L)])
        
        # 解线性系统 P^{t+1} = inv(lhs) * rhs
        new_points = scipy.sparse.linalg.spsolve(lhs, rhs)
        
        points = new_points
        pcd.points = o3d.utility.Vector3dVector(points)
    
    return pcd

def don_pointcloud_gs(gaussians,radius1, radius2):
    """
    Estimate normals of a point cloud
    radius: number of neighbors to consider
    """
    nn = gaussians.knn_idx
    normals = gaussians.get_smallest_axis()
    neighbors = nn[:, 1:]
    normals_neighbor = normals[neighbors]
    normals_r1 = normals_neighbor[:, 1:radius1,:].mean(dim=-1)
    normals_r2 = normals_neighbor[:, radius1:radius2,:].mean(dim=-1)
    return 0.5 * (normals_r1 - normals_r2)

def don_pointcloud(points,radius1=0.0001, radius2=0.001, knn1=10, knn2=100, method='radius'):
    # difference of normals operator
    # use open3d to compute normals
    if method == 'knn':
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn1))
        normals_r1 = np.asarray(pcd.normals).copy()
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn2))
        normals_r2 = np.asarray(pcd.normals)
    elif method == 'radius':
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius1))
        normals_r1 = np.asarray(pcd.normals).copy()
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius2))
        normals_r2 = np.asarray(pcd.normals)
    else:
        raise ValueError("Method not supported")
    result = 0.5 * (normals_r1 - normals_r2)
    norm_diff = np.linalg.norm(result, axis=1)
    return norm_diff

def don_func(gaussian,radius1, radius2,threshold, knn1,knn2,method='radius',vis_don_pointcloud=False):
    # split leaf and branch  based on point cloud
    xyz = gaussian._xyz.detach().cpu().numpy()
    don = don_pointcloud(gaussian, radius1=radius1,radius2=radius2,knn1=10,knn2=1000,method=method)
    points_branch = xyz[don >threshold]
    points_leaf = xyz[don <threshold]
    # save branch points
    pcd_branch = o3d.geometry.PointCloud()
    pcd_branch.points = o3d.utility.Vector3dVector(points_branch)
    pcd_leaf = o3d.geometry.PointCloud()
    pcd_leaf.points = o3d.utility.Vector3dVector(points_leaf)
    
    # save don colored point cloud 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # color by don
    norm_don = (don - don.min()) / (don.max() - don.min())
    colors = cm.viridis(norm_don)[:, :3]  # 使用 matplotlib colormap
    if vis_don_pointcloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(f"don_colored_{radius1}_{radius2}.ply", pcd)
        o3d.io.write_point_cloud("output/ficus/max_5000/point_cloud/iteration_7000/branch.ply", pcd_branch)
        o3d.io.write_point_cloud("output/ficus/max_5000/point_cloud/iteration_7000/leaf.ply", pcd_leaf)

def fit_cylinder_ransac(points,  eps=0.005,min_samples=5,save_ply=False): 
    from sklearn.cluster import DBSCAN

    # Dummy logic: let's just run DBSCAN to group roughly linear segments (can be seen as 'branches')
    # (0.03,5) for plant4 | (0.005,5) for ficus | (0.05,5) for plant1

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points) # 0.005,5
    labels = clustering.labels_
    
    # Assign color by label
    colors = np.random.rand(len(set(labels)), 3)
    point_colors = np.array([colors[l] if l != -1 else [0.8, 0.8, 0.8] for l in labels])

    # Save to PLY
    if save_ply:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(point_colors)
        o3d.io.write_point_cloud("dbscan_segmentation_ficus.ply", pcd)
        print("Saved: dbscan_segmentation.ply")
    
    # add leaf branch filter
    leaf_color = [0.0, 1.0, 0.0]     # green
    branch_color = [1.0, 0.0, 0.0]   # red
    noise_color = [0.7, 0.7, 0.7]    # gray
    unique_labels = set(labels)
    label_leaf = []
    label_branch = []
    for label in unique_labels:
        if label == -1:
            # Noise
            point_colors[labels == label] = noise_color
            continue

        cluster_points = points[labels == label]

        if len(cluster_points) < 100:
            point_colors[labels == label] = noise_color
            continue

        if is_leaf(cluster_points):
            point_colors[labels == label] = leaf_color
            label_leaf.append(label)
        else:
            point_colors[labels == label] = branch_color
            label_branch.append(label)
    if save_ply:
        pcd.colors = o3d.utility.Vector3dVector(point_colors)
        o3d.io.write_point_cloud("dbscan_segment_leaf_branch_ficus.ply", pcd)
    # return three labels
    return label_leaf, label_branch, labels

def z_axis_to_vector_rotation(target_vector,target: Literal['gs', 'cylinder']):
    """Compute rotation that aligns [0, 0, 1] to target_vector"""
    target_vector = target_vector / np.linalg.norm(target_vector)
    if target == 'gs':
        z_axis = np.array([1, 0, 0])
    elif target == 'cylinder':
        z_axis = np.array([0, 0, 1])
    else:
        raise ValueError("target must be 'gs' or 'cylinder'")
    v = np.cross(z_axis, target_vector)
    c = np.dot(z_axis, target_vector)
    if np.isclose(c, 1.0):  # Already aligned
        return np.eye(3)
    if np.isclose(c, -1.0):  # Opposite
        return R.from_rotvec(np.pi * np.array([1, 0, 0])).as_matrix()
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    rot_matrix = np.eye(3) + vx + vx @ vx * (1 / (1 + c))
    return rot_matrix

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

def estimate_gs_para_from_cluster(xyz,test_flag=False):
    cov = np.cov(xyz.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # width = 3 * np.sqrt(eigvals[0])
    # height = 3 * np.sqrt(eigvals[1])
    width = 3 * np.sqrt(eigvals[1])
    height = 3 * np.sqrt(eigvals[0])
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    center = np.mean(xyz, axis=0)    
    major_axis = eigvecs[:, 0]
    minor_axis = eigvecs[:, 1]
    normal = eigvecs[:, 2]
    # rot_gs = np.roll(rot_gs,1)
    rot_matrix_cylinder = z_axis_to_vector_rotation(major_axis,target='cylinder') 
    rot_matrix_gs = z_axis_to_vector_rotation(major_axis, target='gs')
    # rot_matrix_disk = z_axis_to_vector_rotation(normal, target='cylinder')
    rot_matrix_disk = eigvecs
    rot_gs = R.from_matrix(rot_matrix_gs).as_quat()
    rot_gs = np.roll(rot_gs, 1)  # [x,y,z,w] - > [w,x,y,z]

    # for cylinder main axis
    scale = np.sqrt(eigvals).clip(min=0.01)
    if test_flag:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=5, alpha=0.6)
        length =0.1
        ax.quiver(center[0], center[1], center[2], major_axis[0], major_axis[1], major_axis[2], length=length, color='r', label='Major Axis')
        ax.quiver(center[0], center[1], center[2], minor_axis[0], minor_axis[1], minor_axis[2], length=length, color='g', label='Minor Axis')
        ax.quiver(center[0], center[1], center[2], normal[0], normal[1], normal[2], length=length, color='b', label='Normal')
        ax.set_title("PCA Eigenvectors")
        ax.legend()
        plt.show()
    return center,rot_gs,scale, rot_matrix_cylinder,rot_matrix_disk

def branch_to_cylinder(branch_points,branch_positions,branch_scales, branch_rotations,filename="cylinder_branch_init.ply",save_flag=False):
    cylinder_meshes = []

    for pos, scale, rot_matrix in zip(branch_positions, branch_scales, branch_rotations):
        # 默认方向：cylinder 沿 z 轴生成
        # height = np.sqrt(scale[0]) * 2  # 主轴长度为 height
        # radius = np.sqrt(scale[1]) * 2  # 横截面尺寸，可以调整
        height = 3*scale[0]
        radius = scale[1]
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=20, split=4)
        # R,t
        # rot_matrix = R.from_quat(quat).as_matrix()
        cylinder.rotate(rot_matrix, center=(0, 0, 0))
        cylinder.translate(pos, relative=False)
        cylinder.paint_uniform_color([0.6, 0.3, 0.0])  # 木头色（棕色）

        cylinder_meshes.append(cylinder)
    # 合并所有 mesh 并保存
    mesh_all = o3d.geometry.TriangleMesh()
    for m in cylinder_meshes:
        mesh_all += m
    # add branch points
    branch_pcd = o3d.geometry.PointCloud()
    branch_pcd.points = o3d.utility.Vector3dVector(branch_points)
    branch_pcd.paint_uniform_color([1.0, 1.0, 0.0]) 
    if save_flag:
        o3d.io.write_triangle_mesh(filename, mesh_all)
        o3d.io.write_point_cloud("branch_points.ply", branch_pcd)
        print(f"[✓] Saved {filename} with {len(branch_positions)} cylinders.")
    return cylinder_meshes

def convert_gs_rot_to_cylinder(rot_gs_quat):
    R_gs = quaternion_to_matrix(rot_gs_quat)
    major_axis = R_gs[:, 0]
    rot_cylinder = z_axis_to_vector_rotation_torch(major_axis, target='cylinder')
    return rot_cylinder

def convert_gs_rot_to_disk(rot_gs_quat):
    R_gs = quaternion_to_matrix(rot_gs_quat)
    normal_axis = R_gs[:, 2]
    rot_disk = z_axis_to_vector_rotation_torch(normal_axis, target='cylinder')
    return rot_disk
        

def leaf_to_disk(leaf_positions, leaf_scales, leaf_rotations, filename="leaf_disk_init.ply",resolution=50,save_flag=False):
    leaf_disk = []
    for center, scale, rot_matrix in zip(leaf_positions, leaf_scales, leaf_rotations):
        a = 2 * scale[0]
        b =   scale[1]
        theta = np.linspace(0, 2 * np.pi, resolution)
        x = a * np.cos(theta)
        y = b * np.sin(theta)
        z = np.zeros_like(x)
        ellipse_points = np.stack([x, y, z], axis=1)  # (N, 3)
        vertices = []
        triangles = []
        vertices.append([0, 0, 0])  
        for pt in ellipse_points:
            vertices.append(pt.tolist())
        for i in range(1, resolution):
            triangles.append([0, i, i + 1])
        triangles.append([0, resolution, 1])  # wrap around
        vertices = np.array(vertices)
        vertices = vertices @ rot_matrix.T
        vertices += center  # 平移

        # build mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.0, 0.8, 0.0])  # 绿色叶片
        leaf_disk.append(mesh)

    if save_flag:
        mesh_all = o3d.geometry.TriangleMesh()
        for m in leaf_disk:
            mesh_all += m
        o3d.io.write_triangle_mesh(filename, mesh_all)
    print(f"[✓] Saved {filename} with {len(leaf_disk)} disks.")
    return leaf_disk

def leaf_to_disk_param(stpr_pos,stpr_scales,stpr_rots,convert_mesh=False):
    centers = stpr_pos
    axes_x = 2*stpr_scales[:,0]
    axes_y = stpr_scales[:,1]
    rot_disk = convert_gs_rot_to_disk(stpr_rots)
    leaf_disk_param = {
        'centers': centers,
        'axes_x': axes_x,
        'axes_y': axes_y,
        'rot_matrix': rot_disk
    }
    if convert_mesh:
        cemter_np = centers.detach().cpu().numpy()
        scale_np = stpr_scales.detach().cpu().numpy()
        rot_matrix_np = rot_disk.detach().cpu().numpy()
        leaf_disks= leaf_to_disk(cemter_np, scale_np, rot_matrix_np, filename="leaf_disk_vis.ply",resolution=50,save_flag=True)
    return leaf_disk_param,leaf_disks

def gs_to_cylinder_distance(
        xyz: torch.Tensor,                  # (M,3) appGs 位置
        parent: torch.LongTensor,           # (M,)  属于哪个 StPr
        cyl_param  # 'center','axis','radius','half_len'
) -> torch.Tensor:                          # 返回 (M,) 距离
    C = cyl_param['center'][parent]         # (M,3)
    u = cyl_param['axis'][parent]           # (M,3) 已归一化
    r = cyl_param['radius'][parent]         # (M,)
    h = cyl_param['half_len'][parent]       # (M,)
    C = C.squeeze(1)  
    u = u.squeeze(1)  
    v   = xyz - C                           # (M,3)
    t   = torch.sum(v * u, dim=-1)          # 投影到轴坐标
    t_c = torch.clamp(t.unsqueeze(1), -h, h)             # 夹到端盖内
    P_a = C + t_c * u         # 轴上最近点
    d_side = torch.norm(xyz - P_a, dim=-1).unsqueeze(1) - r
    d_side = torch.clamp(d_side, min=0.0)   # 侧壁距离 (负值→0)

    # 端盖外侧的距离
    cap_mask = (t.abs().unsqueeze(1) > h)
    d_cap = torch.sqrt(
        d_side[cap_mask]**2 + (t.abs().unsqueeze(1)[cap_mask]-h[cap_mask])**2
    )
    d_side[cap_mask] = d_cap
    return d_side.squeeze(1)                           # (M,)


# ────────────────────────────────────────────────────────────
# 2. GS ↔ Elliptic Disk 距离
# ─────────────────────────────────────────────────────────────
def gs_to_disk_distance(
        xyz: torch.Tensor,                 # (M,3)
        parent: torch.LongTensor,          # (M,)
        disk_param# 'center','normal','a','b'
) -> torch.Tensor:                         # (M,)
    C = disk_param['center'][parent].squeeze(1)      # (M,3)
    n = disk_param['normal'][parent].squeeze(1) 
    a = disk_param['a'][parent].squeeze(1)           # (M,)
    b = disk_param['b'][parent].squeeze(1)            # (M,)
    v = xyz - C
    d_plane = torch.abs(torch.sum(v * n, dim=-1))           # |z|

    # 构平面的局部坐标基 e1,e2
    tmp = torch.tensor([1.,0.,0.], device=xyz.device).expand_as(n)
    e1 = F.normalize(torch.cross(n, tmp), dim=-1)
    # 当 n≈x 轴时，用 y 轴生成
    bad = torch.isnan(e1).any(dim=-1)
    if bad.any():
        tmp2 = torch.tensor([0.,1.,0.], device=xyz.device).expand_as(n[bad])
        e1[bad] = F.normalize(torch.cross(n[bad], tmp2), dim=-1)
    e2 = torch.cross(n, e1)

    # 坐标 (x',y') in disk plane
    x_coord = torch.sum(v * e1, dim=-1)
    y_coord = torch.sum(v * e2, dim=-1)

    r_val = torch.sqrt((x_coord/a)**2 + (y_coord/b)**2)     # 椭圆极半径
    # 到椭圆边的径向距离
    d_edge = torch.sqrt( (x_coord - a*r_val.reciprocal())**2 +
                         (y_coord - b*r_val.reciprocal())**2 )

    d = torch.where(r_val <= 1.0,
                    d_plane,
                    torch.sqrt(d_plane**2 + d_edge**2))
    return d

def is_leaf(points, flatness_thresh=0.1, anisotropy_thresh=0.95): # 0.1, 0.8
    # record the anisotropy and flatness
    pca = PCA(n_components=3)
    pca.fit(points)
    eigvals = pca.explained_variance_

    # flatness ratio: z-direction variance vs major axis
    flatness = eigvals[2] / (eigvals[0] + 1e-6)
    anisotropy = (eigvals[0] - eigvals[1]) / (eigvals[0] + 1e-6)
    # print(f"Anisotropy: {anisotropy}, Flatness: {flatness}")
    
    if anisotropy < anisotropy_thresh : # and flatness < flatness_thresh
        return True # leaf-like
    return False    # branch-like

def _torch_to_numpy(x: torch.Tensor):
    return x.detach().cpu().numpy()

def align_Z_to_u(u: torch.Tensor) -> torch.Tensor:
    """
    输入单位向量 u (…,3)，返回 3×3 旋转矩阵，把局部 (0,0,1) 转到 u。
    """
    z = torch.tensor([0.,0.,1.], device=u.device, dtype=u.dtype)
    v = torch.cross(z, u)
    c = torch.sum(z * u, dim=-1, keepdim=True)        # cosθ
    s = torch.norm(v, dim=-1, keepdim=True)           # sinθ
    # R = I + [v]_x + [v]_x^2 * ((1-c)/s^2)
    vx = torch.zeros(u.shape[:-1] + (3,3), device=u.device, dtype=u.dtype)
    vx[..., 0,1], vx[..., 0,2] = -v[...,2],  v[...,1]
    vx[..., 1,0], vx[..., 1,2] =  v[...,2], -v[...,0]
    vx[..., 2,0], vx[..., 2,1] = -v[...,1],  v[...,0]
    eye = torch.eye(3, device=u.device, dtype=u.dtype).expand_as(vx)
    R = eye + vx + vx @ vx * ((1-c)/ (s**2 + 1e-8))
    # 当 u≈±z 时，v≈0，算法退化；手动处理：
    parallel = s.squeeze(-1) < 1e-6
    R[parallel & (c.squeeze(-1) > 0)] = eye[0]        # 同向: I
    R[parallel & (c.squeeze(-1) < 0)] = torch.diag(torch.tensor([1,-1,-1],device=u.device,dtype=u.dtype)) # 反向: 180°绕X
    return R

def stpr_to_cylinder(p, S, R,save_flag=True,resolution=32):
    rot_matrix = quaternion_to_matrix(R)  # (N,3,3)
    r = S[:,1]   # (N,)
    h = 3* S[:, 0]                        # 半长 (从中心到端盖)
    u_save = rot_matrix[:, :, 2]                     # 轴向单位向量 (N,3)

    if save_flag:
        mesh_all = o3d.geometry.TriangleMesh()
        mesh_all_list = []
        for i in range(p.shape[0]):
            mesh = o3d.geometry.TriangleMesh.create_cylinder(
                radius=float(r[i]), height=float(h[i]),
                resolution=resolution, split=20)
            u = rot_matrix[i,:,0]                            # 目标主轴
            R_align = align_Z_to_u(u)
            mesh.rotate(_torch_to_numpy(R_align), center=(0, 0, 0))
            mesh.translate(_torch_to_numpy(p[i]))
            mesh_all = mesh_all + mesh
            mesh_all_list.append(mesh)
        #o3d.io.write_triangle_mesh(f'branch_cylinder_vis.ply', mesh_all)
        return {'center': p, 'axis': u_save, 'radius': r, 'half_len': h}, mesh_all, mesh_all_list
    else:
        return {'center': p, 'axis': u_save, 'radius': r, 'half_len': h}, None,None

def stpr_to_disk(p, S, R, save_flag=False,resolution=32):
    rot_matrix = quaternion_to_matrix(R)  # (N,3,3)
    a = 2 * S[:, 0]
    b = S[:, 1]   # (N,)
    u,v,n = rot_matrix[:, :, 0], rot_matrix[:, :, 1], rot_matrix[:, :, 2] 
    if save_flag:
        mesh_all = o3d.geometry.TriangleMesh()
        mesh_all_list = []
        # 预生成圆周角度 (CPU 张量即可)
        theta = torch.linspace(0, 2*torch.pi, steps=resolution+1)[:-1]  # (R,)
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)               # (R,)

        for i in range(p.shape[0]):
            # 3.1 椭圆周圈顶点 (R,3)
            x = a[i] * cos_t.to(a.device)                         
            y = b[i] * sin_t.to(a.device)                        
            verts_ring = (p[i]                                          
                          + x.unsqueeze(1) * u[i]                        
                          + y.unsqueeze(1) * v[i])                      

            # 3.2 顶点堆叠 + fan triangulation
            verts = torch.cat([p[i].unsqueeze(0), verts_ring], dim=0)    # (R+1,3)
            tri_idx = torch.stack([                                       # (R,3)
                torch.zeros(resolution, dtype=torch.int64, device=p.device),       # center idx 0
                torch.arange(1, resolution+1, device=p.device) % resolution + 1,
                torch.arange(1, resolution+1, device=p.device)
            ], dim=1)

            # 3.3 转成 Open3D Mesh
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices  = o3d.utility.Vector3dVector(_torch_to_numpy(verts))
            mesh.triangles = o3d.utility.Vector3iVector(_torch_to_numpy(tri_idx))
            mesh.compute_vertex_normals()
            mesh_all_list.append(mesh)
            mesh_all += mesh
        #TODO: fix rotation bug
        o3d.io.write_triangle_mesh('leaf_disk_vis.ply', mesh_all)
        return {'center': p, 'normal': n, 'a': a, 'b': b}, mesh_all,mesh_all_list
    else:
        return {'center': p, 'normal': u, 'a': a, 'b': b}, None, None

def build_edge(top, bottom,save_edge=False):
        # save a mesh of edge
        N = top.size(0)
        verts = torch.empty((2*N, 3), dtype=torch.float32, device=top.device)
        verts[0::2] = top
        verts[1::2] = bottom
        verts_np = verts.detach().cpu().numpy()
        edges_np = np.arange(2*N, dtype=np.int32).reshape(-1, 2)
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(verts_np)
        ls.lines  = o3d.utility.Vector2iVector(edges_np)
        idx = torch.arange(N, dtype=torch.long, device=top.device)


        # ---------- 3‑B  optional: save PLY with vertex+edge -------------------
        if save_edge:
            v_struct = np.empty(2*N, dtype=[('x','f4'),('y','f4'),('z','f4')])
            v_struct['x'], v_struct['y'], v_struct['z'] = verts_np.T
            el_v = PlyElement.describe(v_struct, 'vertex')

            e_struct = np.empty(N, dtype=[('vertex1','u4'),('vertex2','u4')])
            e_struct['vertex1'] = edges_np[:,0]
            e_struct['vertex2'] = edges_np[:,1]
            el_e = PlyElement.describe(e_struct, 'edge')

            PlyData([el_v, el_e], text=True).write('edge.ply')
        return verts, 2*idx, 2*idx +1

def build_mst_from_endpoints(top, bottom, k:int=3):
    top = top.detach().cpu().numpy()
    bottom = bottom.detach().cpu().numpy()
    N = top.shape[0]
    points = np.empty((2*N, 3), dtype=np.float32)
    points[0::2] = top
    points[1::2] = bottom    
    M = points.shape[0]
    row, col, data = [], [], []

    # (1)  zero‑weight internal edges
    idx = np.arange(N, dtype=np.int32)
    row.extend(2*idx)           ; col.extend(2*idx+1)
    data.extend(np.zeros(N))    # weight 0
    #  symmetric entry
    row.extend(2*idx+1)         ; col.extend(2*idx)
    data.extend(np.zeros(N))

    # (2)  k‑NN edges  (Euclidean distance)
    tree = cKDTree(points)
    dists, neigh = tree.query(points, k=k+1)     # first neighbour is itself

    for i in range(M):
        for j, d in zip(neigh[i, 1:], dists[i, 1:]):   # skip self
            row.append(i);  col.append(j);  data.append(d)
            # symmetric entry
            row.append(j);  col.append(i);  data.append(d)

    # ---------- build symmetric CSR adjacency ----------
    A = csr_matrix((data, (row, col)), shape=(M, M))

    # ---------- Minimum Spanning Tree ----------
    T_csr  = minimum_spanning_tree(A)            # still CSR  (M×M)
    mst_edges = np.vstack(T_csr.nonzero()).T     # (M-1, 2)  index pairs
    mst_w     = T_csr.data  
    return mst_edges, points

def save_mst_ply(points, edges, path='mst.ply'):
    from plyfile import PlyElement, PlyData
    v = np.empty(points.shape[0], dtype=[('x','f4'),('y','f4'),('z','f4')])
    v['x'], v['y'], v['z'] = points.T
    e = np.empty(edges.shape[0], dtype=[('vertex1','u4'),('vertex2','u4')])
    e['vertex1'] = edges[:,0];  e['vertex2'] = edges[:,1]
    PlyData([PlyElement.describe(v,'vertex'),
             PlyElement.describe(e,'edge')], text=True).write(path)
if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse



    
    # args = parser.parse_args(sys.argv[1:])
    # args.save_iterations.append(args.iterations)
    
    # print("Optimizing " + args.model_path)

    # # Initialize system state (RNG)
    # safe_state(args.quiet)
    # torch.cuda.set_device(args.gpu)
    # device = torch.device(f"cuda:{args.gpu}")
    # print(f"Using device {device}")
    # args.device = device
    # gaussian_file = 'output/plant3_3dgs/point_cloud/iteration_3000/point_cloud.ply'
    # dataset = lp.extract(args)
    # opt = op.extract(args)
    # pipeline = pp.extract(args)
    # gaussian = GaussianModel(dataset.sh_degree, opt.optimizer_type, args.device)
    # gaussian.load_ply(gaussian_file)
    # gaussian.knn_to_track = 32
    # gaussian.reset_neighbors()
    # xyz = gaussian._xyz.detach().cpu().numpy()
    

    # """ don operator """
    # # radius1 = 0.0001
    # # radius2 = 0.1
    # # threshold = 0.55
    # # don_func(radius1, radius2,threshold, knn1=10,knn2=1000,method='radius')

    # """ RANSAC clustering """
    # fit_cylinder_ransac(xyz, num_iterations=1000, distance_threshold=0.01)
    # # based on gaussian parameter
    # # cov_matrices = gaussian.get_covariance(return_full=True)
    # # eigvals = torch.linalg.eigvalsh(cov_matrices)  # (N, 3)
    # # sigma_max, sigma_min = eigvals[:, -1], eigvals[:, 0]
    # # anisotropy = (sigma_max - sigma_min) / (sigma_max + 1e-8)
    # # normals = gaussian.get_smallest_axis()
    # pass
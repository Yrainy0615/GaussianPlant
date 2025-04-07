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

def fit_cylinder_ransac(points,  eps=0.03,min_samples=5,save_ply=False):
    from sklearn.cluster import DBSCAN

    # Dummy logic: let's just run DBSCAN to group roughly linear segments (can be seen as 'branches')
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
        o3d.io.write_point_cloud("dbscan_segmentation_plant3.ply", pcd)
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
        o3d.io.write_point_cloud("dbscan_segment_leaf_branch_plant3.ply", pcd)
    # return three labels
    return label_leaf, label_branch, labels

def z_axis_to_vector_rotation(target_vector):
    """Compute rotation that aligns [0, 0, 1] to target_vector"""
    target_vector = target_vector / np.linalg.norm(target_vector)
    z_axis = np.array([0, 0, 1])
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
    rot_matrix = z_axis_to_vector_rotation(major_axis)
    # Use PCA eigenvectors as orientation
    rot = R.from_matrix(rot_matrix).as_quat()
    scale = np.sqrt(eigvals).clip(min=0.01)
    if test_flag:
        theta_deg = np.degrees(np.arctan2(major_axis[1], major_axis[0]))
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(xyz[:, 0], xyz[:, 1], s=5, alpha=0.6)
        ax.plot(center[0], center[1], 'ro', label='Center')
        ellipse = Ellipse(xy=center[:2], width=2*width, height=2*height,
                        angle=theta_deg, edgecolor='r', facecolor='none', linewidth=2)
        ax.add_patch(ellipse)
        ax.set_aspect('equal')
        plt.legend()
        plt.grid(True)
        plt.savefig("cluster_to_gs.png")
        plt.close()
  
    return center,rot,scale

def branch_to_cylinder(branch_points,branch_positions,branch_scales, branch_rotations,filename="cylinder_branch_init.ply"):
    cylinder_meshes = []

    for pos, scale, quat in zip(branch_positions, branch_scales, branch_rotations):
        # 默认方向：cylinder 沿 z 轴生成
        # height = np.sqrt(scale[0]) * 2  # 主轴长度为 height
        # radius = np.sqrt(scale[1]) * 2  # 横截面尺寸，可以调整
        height = 3*scale[0]
        radius = scale[1]
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=20, split=4)
        # R,t
        rot_matrix = R.from_quat(quat).as_matrix()
        cylinder.rotate(rot_matrix, center=(0, 0, 0))
        cylinder.translate(pos, relative=False)
        cylinder.paint_uniform_color([0.6, 0.3, 0.0])  # 木头色（棕色）

        cylinder_meshes.append(cylinder)

    # 合并所有 mesh 并保存
    mesh_all = cylinder_meshes[0]
    for m in cylinder_meshes[1:]:
        mesh_all += m
    # add branch points
    branch_pcd = o3d.geometry.PointCloud()
    branch_pcd.points = o3d.utility.Vector3dVector(branch_points)
    branch_pcd.paint_uniform_color([1.0, 1.0, 0.0])  # green
    o3d.io.write_triangle_mesh(filename, mesh_all)
    o3d.io.write_point_cloud("branch_points.ply", branch_pcd)
    print(f"[✓] Saved {filename} with {len(branch_positions)} cylinders.")


def leafcluster_to_2dgs(xyz):
    cov = np.cov(xyz.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    width = 3 * np.sqrt(eigvals[0])
    height = 3 * np.sqrt(eigvals[1])
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    center = np.mean(xyz, axis=0)
    major_axis = eigvecs[:, 0]
    minor_axis = eigvecs[:, 1]
    normal = eigvecs[:, 2]
    s1 = 3 * np.sqrt(eigvals[0])
    s2 = 3 * np.sqrt(eigvals[1])
    theta_deg = np.degrees(np.arctan2(major_axis[1], major_axis[0]))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xyz[:, 0], xyz[:, 1], s=5, alpha=0.6)
    ax.plot(center[0], center[1], 'ro', label='Center')
    ellipse = Ellipse(xy=center[:2], width=width, height=height,
                    angle=theta_deg, edgecolor='r', facecolor='none', linewidth=2)
    ax.add_patch(ellipse)
    ax.set_aspect('equal')
    ax.set_title("Leaf Point Cloud and Fitted 2D Gaussian Disk")
    plt.legend()
    plt.grid(True)
    # plt.show()
    # save the figure
    plt.savefig("leaf_pca.png")

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
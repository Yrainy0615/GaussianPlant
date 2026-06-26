import os
import torch
import argparse
from arguments import ModelParams
from scene import Scene
from scene.gaussian_model import GaussianModel
import open3d as o3d
from utils.general_utils import safe_state, get_expon_lr_func, build_rotation
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render, network_gui, render_gsplant
from utils.gs_utils import strpr_to_cylinder, strpr_to_disk,build_mst_from_endpoints,save_mst_ply, gs_to_cylinder_distance,build_mst_from_endpoints, build_polyline_graph, compute_node_radius_from_edge_radius
import sys
import networkx as nx
import numpy as np
from utils.loss_utils import laplacian_smooth_loss, loss_endpoints
from utils.visualization import  make_tube_mesh_from_graph
from utils.pytorch3d_compat import chamfer_distance

def infer_strpr_radius(strpr_scale: torch.Tensor,
                       mode: str = "geom",
                       eps: float = 1e-6) -> torch.Tensor:
    """
    strpr_scale: (S,3) 其中 [:,0]=轴向半长h, [:,1],[: ,2]=截面两个半径尺度
    mode: 'geom' (几何平均) | 'mean' (算术平均) | 'min' (取较小)
    return: (S,) 每个 \st 的等效半径
    """
    r1 = strpr_scale[:, 1].abs()
    r2 = strpr_scale[:, 2].abs()
    if mode == "geom":
        r = (r1 * r2).clamp_min(0).sqrt()
    elif mode == "mean":
        r = 0.5 * (r1 + r2)
    elif mode == "max":
        r = torch.maximum(r1, r2)
    elif mode == "min":
        r = torch.minimum(r1, r2)
    else:
        raise ValueError("mode must be 'geom' | 'mean' | 'min'")
    return r.clamp_min(eps)


def edge_radius_with_pairs_for_endpoint_graph(
    tree_edges: torch.Tensor,      # (E,2) on p=[top,bottom]
    strpr_pairs: torch.Tensor,     # (S,2) = [[0,1],[2,3],...]
    n_strpr: int,
    strpr_scale: torch.Tensor,     # (S,3)
    mode: str = "geom",
) -> torch.Tensor:
    """
    规则：
      - 若 edge ∈ strpr_pairs（内部边），半径=对应 \st 半径
      - 否则（外部边），半径=两端 \st 半径的均值
    """

    E = tree_edges.shape[0]
    strpr_r = infer_strpr_radius(strpr_scale, mode=mode)  # (S,)

    # endpoint -> 所属 \st（0,1 -> 0; 2,3 -> 1; ...），用整除更稳
    endpoint_sid = (torch.arange(2*n_strpr) // 2).long()  # (2S,)

    u, v = torch.tensor(tree_edges[:, 0]), torch.tensor(tree_edges[:, 1])
    su, sv = endpoint_sid[u], endpoint_sid[v]
    ru, rv = strpr_r[su], strpr_r[sv]

    # --- 判断是否为内部对：把 (i,j) 排序后做哈希键，O(E) ---
    # 先把 pairs 按小->大排序
    pp = torch.sort(strpr_pairs.long(), dim=1).values
    pe = torch.sort(torch.stack([u, v], dim=1).to(strpr_pairs.device), dim=1).values

    # 哈希：key = a * K + b，K 取 > max_index 的常数；这里用 2*n_strpr
    K = 2 * n_strpr
    pair_keys  = pp[:, 0] * K + pp[:, 1]        # (S,)
    edge_keys  = pe[:, 0] * K + pe[:, 1]         # (E,)

    # membership：torch.isin 在 2.0+ 有；若无可转 numpy 用 np.isin
    same = torch.isin(edge_keys, pair_keys)      # (E,) bool，True 表示内部边

    # 半径：内部=所属 \st 的半径（su==sv，对应同一个 \st），外部=均值
    # su 与 sv 在内部边时相等（因为 pair 必然是(2i,2i+1)）
    edge_r = torch.where(same, ru, 0.5 * (ru + rv))
    return edge_r  # (E,)

def gs_to_cylinder(pos, scale, rot, save_path, iteration=0,create_mesh=True):
    cylinder_params, mesh_cylinder_all, mesh_cylinder_list = strpr_to_cylinder(pos,
                                                                               scale,
                                                            rot,
                                                            iteration,
                                                            create_mesh=create_mesh,
                                                            save_path =save_path,
                                                            resolution=32)
def gs_to_graph(strprs, save_path,save_flag=True):
    center = strprs.get_xyz
    strpr_scale = strprs.get_scaling
    rotation = build_rotation(strprs.get_rotation)
    u = rotation[:,:, 0]
    h = strprs.get_scaling[:, 0] 
    top, bottom = center + h[:, None] * u, center - h[:, None] * u
    n_strpr = center.shape[0]
    # mst_edges, points = build_mst_from_endpoints(top, bottom)
    p = torch.stack((top, bottom), dim=1).reshape(-1,3)   
    strpr_pairs = torch.stack([
        torch.arange(0, 2*n_strpr, 2, device=p.device),
        torch.arange(1, 2*n_strpr, 2, device=p.device)
    ], dim=1)
    tree_edges = build_mst_from_endpoints(p, strpr_pairs, k=8)
    save_mst_ply( p, tree_edges, save_path)
    # center points graph
    # save_path_center = save_path.replace(".ply", "_center.ply")
    # tree_edges = build_mst_from_endpoints(center, strpr_pairs=None, k=8)
    # save_mst_ply( center, tree_edges, save_path_center)

    # radius edges
    edge_r = edge_radius_with_pairs_for_endpoint_graph(tree_edges, strpr_pairs,n_strpr, strpr_scale, mode="geom")  # (E,)


    node_r = compute_node_radius_from_edge_radius(
        num_nodes=p.shape[0],
        edges=tree_edges,
        edge_radii=edge_r,
        reduce="max"  # 或 "mean"
    )
    # prune_leaf_strpr_by_radius_growth(tree_edges, node_r )

    return p,tree_edges, node_r

def remove_gs(strpr):
    opacity_threshold = 0.2
    scale_threshold = 0.0001 # 0.001
    mask = (strpr.get_opacity.mean(dim=1) > opacity_threshold) & (strpr.get_scaling[:,0] > scale_threshold)
    prune_mask = ~mask
    strpr.prune_points(prune_mask)   
    strpr.save_ply(args.strpr_path.replace(".ply", "_pruned.ply"))

def remove_branch_from_all(strpr_branch, strpr_all):
    xyz_all = strpr_all.get_xyz
    xyz_branch = strpr_branch.get_xyz
    # prune mask = branch points in all strprs
    knn = torch.cdist(xyz_branch, xyz_all)  # (N_branch, N_all)
    nn = torch.argmin(knn, dim=1)  # (N_branch,)
    prune_mask = torch.zeros(xyz_all.shape[0], dtype=torch.bool, device=xyz_all.device)
    prune_mask[nn] = True
    strpr_all.prune_points(prune_mask)

def strpr_to_surface(leaf_strpr, branch_strpr):
    # leaf surface
    leaf_path = args.strpr_path.replace('.ply', '_leaf.ply')
    _, _, mesh_list = strpr_to_cylinder(pos=leaf_strpr._xyz, 
                    S=leaf_strpr.get_scaling, 
                    R=build_rotation(leaf_strpr.get_rotation), 
                    save_path=leaf_path, 
                    create_mesh=True,
                    iteration=0)
    # branch surface
    branch_path = args.strpr_path.replace('.ply', '_branch.ply')
    _, _, mesh_list = strpr_to_cylinder(pos=branch_strpr._xyz, 
                    S=branch_strpr.get_scaling, 
                    R=build_rotation(branch_strpr.get_rotation), 
                    save_path=branch_path, 
                    create_mesh=True,
                    iteration=0)

def get_gt_branch_points(args):
    appgas = GaussianModel(3, args.device, args.model_path)
    appgas.load_ply(args.appgas_path)
    gt_densepoints = o3d.io.read_point_cloud(args.pcd_path)
    gt_densepoints = torch.from_numpy(np.asarray(gt_densepoints.points)).float().to(appgas.device)

    # get branch points by chamder distance 
    appgas_points = appgas.get_xyz
    dists = torch.cdist(gt_densepoints.to(appgas_points.device),appgas_points)  # (N_appgas, N_gt)
    min_dists, _ = torch.min(dists, dim=1)  # (N_appgas,)
    threshold = 0.01 # 0.01
    branch_mask = min_dists < threshold
    branch_points = gt_densepoints[branch_mask]
    branch_path = args.pcd_path.replace('.ply', '_branch_dense.ply')
    branch_pd = o3d.geometry.PointCloud()
    branch_pd.points = o3d.utility.Vector3dVector(branch_points.detach().cpu().numpy())
    o3d.io.write_point_cloud(branch_path, branch_pd)

def get_branch_from_strpr(gaussians,strpr):
    gaussians.appgas = gaussians
    gaussians.strprs = strpr
    gaussians.update_nn_between_appgas_and_stprs()

    disk_params, cylinder_params, mesh_cylinder_list, mesh_disk_list = gaussians.build_primitive_surface(compute_all=True, create_mesh=False)
    nn = gaussians.nn_stpr_appgas.squeeze(1)
    dis_cylinder = gs_to_cylinder_distance(gaussians._xyz, nn, cylinder_params)
    threshold = 0.05
    branch_mask = dis_cylinder < threshold
    branch_points = gaussians.appgas.get_xyz[branch_mask]
    return branch_points

def appgas_init_from_strpr(gaussians,strpr, branch_strpr):
    gaussians.strprs = strpr
    # branch mask is branch points which in strpr
    knn = torch.cdist(branch_strpr._xyz, strpr._xyz)  # (N_branch, N_all)
    nn = torch.argmin(knn, dim=1)  # (N_branch,)
    branch_mask = torch.zeros(strpr._xyz.shape[0], dtype=torch.bool, device=strpr._xyz.device)
    branch_mask[nn] = True
    appgas = gaussians.build_appgas_from_stprs(num_sample=100, use_pretrain=False,branch_mask=branch_mask)

def main(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    # scene = Scene(dataset, gaussians,load_iteration=checkpoint)
    strpr = GaussianModel(dataset.sh_degree, args.device, args.model_path)
    strpr.load_ply(args.strpr_path)
    strpr.training_setup(opt)
    if args.pcd_path is not None:
        gaussians = GaussianModel(dataset.sh_degree, args.device, args.model_path)
        gaussians.load_ply(args.pcd_path)
    if args.strpr_branch is not None:
        branch_strpr = GaussianModel(dataset.sh_degree, args.device, args.model_path)
        branch_strpr.load_ply(args.strpr_branch)
    # appgas_init_from_strpr(gaussians,strpr,branch_strpr)
    # remove strpr with small scale or low opacity
    opacity_threshold = 0.2 
    scale_threshold = 0.0001 # 0.001
    prune_mask = (strpr.get_opacity <= opacity_threshold).squeeze(1) | (strpr.get_scaling[:,0] <= scale_threshold)
    strpr.prune_points(prune_mask)
    # remove_gs(strpr)

    mst_path = os.path.join(args.model_path, "branch_mst.ply")
    # mst graph
    p,tree_edges, node_r= gs_to_graph(strpr, mst_path)
    poly_graph = build_polyline_graph(
        points=p.detach().cpu().numpy(),
        edges=tree_edges,
        node_radius=node_r.astype(np.float32)
    )
    # save tube mesh
    tube_path = os.path.join(args.model_path, "branch_tube.ply")
    tube = poly_graph.tube(scalars="radius", absolute=True, n_sides=30, capping=True)
    tube.save(tube_path)
    # cylinder
    # cylinder_path = os.path.join(args.model_path, "branch_cylinder.ply")
    # _, _, mesh_list = strpr_to_cylinder(pos=strpr._xyz, 
    #                 S=strpr.get_scaling, 
    #                 R=build_rotation(strpr.get_rotation), 
    #                 save_path=cylinder_path, 
    #                 create_mesh=True,
    #                 iteration=0)
    
    # remove low opacity and small strprs
    # strpr_branch =GaussianModel(dataset.sh_degree, args.device, args.model_path)
    # strpr_branch.load_ply(args.strpr_path.replace('.ply', '_branch.ply'))
    # remove_branch_from_all(strpr_branch, strpr)
    # leaf_path = args.strpr_path.replace('.ply', '_leaf.ply')
    # strpr.save_ply(leaf_path)
    # get_gt_branch_points(args)
    # branch_points = get_branch_from_strpr(gaussians, strpr)
    # branch_path = args.strpr_path.replace('.ply', '_branch_from_strpr.ply')
    # branch_pd = o3d.geometry.PointCloud()
    # branch_pd.points = o3d.utility.Vector3dVector(branch_points.detach().cpu().numpy())
    # o3d.io.write_point_cloud(branch_path, branch_pd)
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[100,300,500,1000,3000,7000,15_000,30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--strpr_path", type=str, default='output/plant1/strpr_branch.ply')
    parser.add_argument("--strpr_branch", type=str, default=None)
    parser.add_argument("--pcd_path", type=str, default=None)
    parser.add_argument("--appgas_path", type=str, default=None)
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7000,15000,30000])
    parser.add_argument("--load_iteration", type=str, default = None)
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')


    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    print(f"Using device {device}")
    args.device = device
    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    main(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.load_iteration, args.debug_from,args)
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from typing import Literal
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH,eval_sh
import math
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import KMeans
import faiss
from utils.pytorch3d_compat import quaternion_to_matrix, quaternion_invert, quaternion_apply, matrix_to_quaternion
from utils.pytorch3d_compat import knn_points
import networkx as nx
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement
import trimesh
from utils.gs_utils import save_labeled_points, estimate_gs_para_from_cluster, build_mst_from_endpoints, strpr_to_disk , strpr_to_cylinder,gs_to_cylinder_distance, gs_to_disk_distance
import time
from utils.loss_utils import mst_loss, loss_endpoints
from utils.pytorch3d_compat import chamfer_distance
import open3d as o3d
import torch.nn.functional as F
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass


def _otsu_threshold(score, bins=256):
    """Otsu's threshold on a 1D score: the value maximising between-class variance, i.e. the
    valley between the (minority, high-score) branch mode and the (majority, low-score) leaf
    mode. Used to auto-calibrate branch_frac per scene without manual tuning."""
    x = score.detach().cpu().numpy().astype(np.float64)
    hist, edges = np.histogram(x, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = hist.sum()
    if total == 0:
        return float(np.median(x))
    w_b = np.cumsum(hist)                                  # count below threshold
    w_f = total - w_b
    sum_b = np.cumsum(hist * centers)
    sum_t = (hist * centers).sum()
    valid = (w_b > 0) & (w_f > 0)
    m_b = np.where(w_b > 0, sum_b / np.maximum(w_b, 1), 0)
    m_f = np.where(w_f > 0, (sum_t - sum_b) / np.maximum(w_f, 1), 0)
    sigma_b = w_b * w_f * (m_b - m_f) ** 2
    sigma_b[~valid] = -1
    return float(centers[int(np.argmax(sigma_b))])


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.label_activation = torch.sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, device=None, save_path: str = None):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.device = device
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.save_path = save_path
        self.knn_to_track = 2
        self.knn_dists = None
        self.knn_idx = None
        self.n_points = None
        self.is_strprs = False
        self.is_appgas = False
        self.appgas = None 
        self.strprs = None
        self.label = None
        self.nn_stpr_appgas = None
        self._semantic_feature = torch.empty(0) 
        self.pca = None  # (1024->768)
        self.pca_low = None # (768 -> dim)
        self.text_feats = None # [2(leaf/stem), dim] 
        

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._semantic_feature, 
            self.label
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D,
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self._semantic_feature,
        self.label) = model_args 
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    @property
    def get_semantic_feature(self):
        return self._semantic_feature 

    @property
    def get_normals(self):
        rot_matrix = build_rotation(self.get_rotation)
        normals = rot_matrix[:,:,2]
        return normals

    def rewrite_semantic_feature(self, x):
        self._semantic_feature = x

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, semantic_feature_size : int, speedup: bool):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        
        if speedup: # speed up for Segmentation
            semantic_feature_size = int(semantic_feature_size/1)
        self._semantic_feature = torch.zeros(fused_point_cloud.shape[0], semantic_feature_size, 1).float().cuda() 
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=self.device)
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=self.device))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)
        self._semantic_feature = nn.Parameter(self._semantic_feature.transpose(1, 2).contiguous().requires_grad_(True))
        self.label = nn.Parameter(torch.zeros((self.get_xyz.shape[0], 1), device=self.device).requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._semantic_feature], 'lr':training_args.semantic_feature_lr, "name": "semantic_feature"},
            {'params': [self.label], 'lr': training_args.label_lr, "name": "label"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        # add semantic feature 
        for i in range(self._semantic_feature.shape[1]*self._semantic_feature.shape[2]):
            l.append('semantic_{}'.format(i))
        for i in range(self.label.shape[1]):
            l.append('label_{}'.format(i))
        return l
    
    def reset_neighbors(self):
        # Compute KNN               
        with torch.no_grad():
            knns = knn_points(self._xyz[None], self._xyz[None], K=self.knn_to_track)
            self.knn_dists = knns.dists[0]
            self.knn_idx = knns.idx[0]

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        label = self.label.detach().cpu().numpy()
        semantic_feature = self._semantic_feature.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, semantic_feature,label), axis=1) 
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_branch_ply(self, path,branch_mask):
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz[branch_mask].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc[branch_mask].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest[branch_mask].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity[branch_mask].detach().cpu().numpy()
        scale = self._scaling[branch_mask].detach().cpu().numpy()
        rotation = self._rotation[branch_mask].detach().cpu().numpy()
        label = self.label[branch_mask].detach().cpu().numpy()
        semantic_feature = self._semantic_feature[branch_mask].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, semantic_feature,label), axis=1) 
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        try:
            plydata = PlyData.read(path.replace('.ply', '_clean.ply'))
        except:
            plydata = PlyData.read(path)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"])),  axis=1)

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        count = sum(1 for name in plydata.elements[0].data.dtype.names if name.startswith("semantic_"))
        semantic_feature = np.stack([np.asarray(plydata.elements[0][f"semantic_{i}"]) for i in range(count)], axis=1) 
        semantic_feature = np.expand_dims(semantic_feature, axis=-1)
        try:
            label = np.stack([np.asarray(plydata.elements[0][f"label_{i}"]) for i in range(count)], axis=1)
        except:
            label = np.zeros((xyz.shape[0], 1), dtype=np.float32)
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=self.device).requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device=self.device).transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device=self.device).transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=self.device).requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=self.device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=self.device).requires_grad_(True))
        self._semantic_feature = nn.Parameter(torch.tensor(semantic_feature, dtype=torch.float, device=self.device).transpose(1, 2).contiguous().requires_grad_(True))
        self.label = nn.Parameter(torch.tensor(label, dtype=torch.float, device=self.device).requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    @torch.no_grad()
    def _mst_edge_cues(self, bc, len_weight, turn_weight):
        """Per-branch-StrPr suspicion from its MST edges: long edge / sharp turn -> floating leaf
        bridged into the tree. Returns a z-scored score (high -> suspect). Optional cue."""
        import networkx as nx
        X = bc.detach().cpu().numpy()
        N = len(X)
        from scipy.spatial import cKDTree
        tb = cKDTree(X)
        pairs = tb.query_pairs(np.median(tb.query(X, k=min(2, N))[0][:, -1]) * 6, output_type='ndarray')
        maxlen = np.zeros(N); turn = np.zeros(N)
        if len(pairs) > 0:
            d = np.linalg.norm(X[pairs[:, 0]] - X[pairs[:, 1]], axis=1)
            G = nx.Graph(); G.add_weighted_edges_from(zip(pairs[:, 0], pairs[:, 1], d))
            T = nx.minimum_spanning_tree(G)
            nbr = {i: [] for i in range(N)}
            for a, b in T.edges():
                w = np.linalg.norm(X[a] - X[b]); nbr[a].append((b, w)); nbr[b].append((a, w))
            for i in range(N):
                if nbr[i]:
                    maxlen[i] = max(w for _, w in nbr[i])
                dirs = [(X[j] - X[i]) / (np.linalg.norm(X[j] - X[i]) + 1e-9) for j, _ in nbr[i]]
                if len(dirs) >= 2:
                    angs = [np.arccos(np.clip(dirs[a] @ dirs[b], -1, 1))
                            for a in range(len(dirs)) for b in range(a + 1, len(dirs))]
                    turn[i] = np.pi - min(angs)
        dev = bc.device
        z = lambda v: (torch.tensor(v, dtype=torch.float, device=dev) - np.mean(v)) / (np.std(v) + 1e-9)
        return len_weight * z(maxlen) + turn_weight * z(turn)

    @torch.no_grad()
    def prune_green_floats(self, green_z=1.0, k=3, scale_min=0.001, leaf_p=0.3):
        """Demote branch StrPr that are clearly GREEN (greenness z-score > green_z) AND isolated
        (above-median distance to nearest branch StrPr). These 'green floating points' slip
        through the joint init (their elongated geometry raised the joint score) but a green
        thing cannot be a woody branch. Mainly a VISUAL cleanup (removes green specks from the
        branch render); may slightly raise Chamfer on clean scenes (some sit near real branches).
        Returns #demoted."""
        p = self.label_activation(self.strprs.label).squeeze()
        branch = (p > 0.5) & (self.strprs.get_scaling.max(dim=1).values > scale_min)
        bidx = torch.nonzero(branch).squeeze(1)
        if bidx.numel() < k + 2:
            return 0
        bc = self.strprs.get_xyz[bidx]
        rgb = 0.2820948 * self.strprs._features_dc[bidx].squeeze(1) + 0.5
        green = rgb[:, 1] - 0.5 * (rgb[:, 0] + rgb[:, 2])
        gz = (green - green.mean()) / (green.std() + 1e-9)
        kk = min(k, bc.shape[0] - 1)
        iso = torch.cdist(bc, bc).topk(kk + 1, largest=False).values[:, kk]
        drop = (gz > green_z) & (iso > iso.median())
        if drop.sum() < 1:
            return 0
        self.strprs.label.data[bidx[drop]] = torch.logit(torch.tensor(leaf_p, device=self.strprs.label.device))
        return int(drop.sum())

    def prune_isolated_branches(self, iso_factor=2.5, k=3, scale_min=0.001, leaf_p=0.3, max_frac=0.25,
                                len_weight=0.0, turn_weight=0.0):
        """Demote spatially-isolated ("floating") branch StrPrs to leaf. True branches form a
        connected tree; mis-classified leaf StrPrs float in space, isolated from other branch
        StrPrs (isolation = distance to the k-th nearest branch StrPr, AUC ~0.84 for detecting
        false branches). ADAPTIVE threshold: demote StrPrs whose isolation > median*iso_factor,
        so clean scenes (no outliers) are barely pruned and messy scenes more -- generalises
        across newplant1-9 (7/9 improve 4-34%, rest ~neutral; iso_factor=2.5). Capped at
        max_frac for safety. Returns #demoted."""
        p = self.label_activation(self.strprs.label).squeeze()
        branch = (p > 0.5) & (self.strprs.get_scaling.max(dim=1).values > scale_min)
        bidx = torch.nonzero(branch).squeeze(1)
        if bidx.numel() < k + 2:
            return 0
        bc = self.strprs.get_xyz[bidx]
        kk = min(k, bc.shape[0] - 1)
        iso = torch.cdist(bc, bc).topk(kk + 1, largest=False).values[:, kk]  # k-th nearest branch
        score = iso
        # Optional MST edge-length / turn-angle cues (off by default: testing showed turn-angle
        # is weak ~0.5-0.64 AUC and hurts scenes where isolation already works; isolation alone
        # is the robust winner, this only helps messy scenes where isolation fails).
        if len_weight > 0 or turn_weight > 0:
            extra = self._mst_edge_cues(bc, len_weight, turn_weight)  # z-scored, high->suspect
            score = (iso - iso.median()) / (iso.std() + 1e-9) + extra
            drop_mask = score > score.median() + iso_factor  # iso_factor as z-threshold here
            if drop_mask.sum() < 1:
                return 0
            if drop_mask.float().mean() > max_frac:
                thr = torch.topk(score, int(bidx.numel() * max_frac)).values.min()
                drop_mask = score >= thr
            self.strprs.label.data[bidx[drop_mask]] = torch.logit(torch.tensor(leaf_p, device=self.strprs.label.device))
            return int(drop_mask.sum())
        drop_mask = iso > iso.median() * iso_factor
        # safety cap: never drop more than max_frac (keep the most isolated within the cap)
        if drop_mask.float().mean() > max_frac:
            thr = torch.topk(iso, int(bidx.numel() * max_frac)).values.min()
            drop_mask = iso >= thr
        if drop_mask.sum() < 1:
            return 0
        self.strprs.label.data[bidx[drop_mask]] = torch.logit(torch.tensor(leaf_p, device=self.strprs.label.device))
        return int(drop_mask.sum())

    def compute_shape_prior(self, k_min=6):
        """Per-StrPr branch logit from the DIMENSIONALITY of its bound AppGS (the binding
        relationship), read in a size-INVARIANT way: linearity (1D tube=branch) vs planarity
        (2D sheet=leaf). This is the geometric-binding label signal, unbiased by primitive size
        (unlike the cylinder/disk *distance* difference, which collapses depending on disk width).

        Returns s_shape in R^S (high -> branch) and a validity mask (StrPr with enough AppGS).
        """
        self.update_nn_between_appgas_and_stprs()
        nn = self.nn_stpr_appgas.squeeze(1)
        xyz = self.appgas._xyz.detach()
        S = self.strprs.get_xyz.shape[0]
        dev = xyz.device
        cnt = torch.zeros(S, device=dev); cnt.index_add_(0, nn, torch.ones(xyz.shape[0], device=dev))
        mean = torch.zeros(S, 3, device=dev); mean.index_add_(0, nn, xyz)
        mean = mean / cnt.clamp_min(1.0).unsqueeze(1)
        # per-point centred outer products, accumulated per StrPr -> covariance
        d = xyz - mean[nn]                                   # (M,3)
        outer = (d.unsqueeze(2) * d.unsqueeze(1)).reshape(-1, 9)  # (M,9)
        cov = torch.zeros(S, 9, device=dev); cov.index_add_(0, nn, outer)
        cov = (cov / cnt.clamp_min(1.0).unsqueeze(1)).reshape(S, 3, 3)
        valid = cnt >= k_min
        ev = torch.linalg.eigvalsh(cov[valid] + 1e-9 * torch.eye(3, device=dev))  # ascending
        l3, l2, l1 = ev[:, 0], ev[:, 1], ev[:, 2]            # l1>=l2>=l3
        denom = l1.clamp_min(1e-9)
        linearity = (l1 - l2) / denom                       # high -> 1D (branch)
        planarity = (l2 - l3) / denom                       # high -> 2D (leaf)
        s = torch.zeros(S, device=dev)
        s[valid] = linearity - planarity                    # in [-1,1], high -> branch
        return s, valid

    def get_semantic_prior(self, visual_feats=None):
        text_feats = F.normalize(self.text_feats, dim=-1)      # [2, dim]
        if visual_feats is None:
            visual_feats = F.normalize(self._semantic_feature.squeeze(1), dim=-1)  # [N, dim]
        else:
            visual_feats = F.normalize(visual_feats.squeeze(1), dim = -1)

        branch_text, leaf_text = text_feats[1], text_feats[0]  # assume index 0=leaf, 1=branch
        # Cosine similarities
        # cos_branch = (visual_feats @ branch_text)              # [N]
        # cos_leaf   = (visual_feats @ leaf_text)                # [N]
        # Margin
        # margin = cos_branch - cos_leaf                         # [N]
        # Convert to probability (branch=1, leaf=0)
        # tau = 0.07  # temperature, adjust if needed
        # p_branch = torch.sigmoid(margin / tau) 
        cos_score = visual_feats @ text_feats.T  # [N, 2]
        probs = F.softmax(cos_score, dim=-1)  # [N, 2]
        p_branch = probs[:, 1:2]  # [N, 1]
        return p_branch

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._semantic_feature = optimizable_tensors["semantic_feature"]
        self.label = optimizable_tensors["label"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        # self.tmp_radii = self.tmp_radii[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_semantic_feature, new_label):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "semantic_feature": new_semantic_feature,
        "label": new_label
    }   

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._semantic_feature = optimizable_tensors["semantic_feature"] 
        self.label = optimizable_tensors["label"]
        # self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device=self.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_semantic_feature = self._semantic_feature[selected_pts_mask].repeat(N,1,1)
        new_label = self.label[selected_pts_mask].repeat(N,1)
        # new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_semantic_feature, new_label) 
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=self.device, dtype=bool)))
        self.prune_points(prune_filter)
        

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_semantic_feature = self._semantic_feature[selected_pts_mask] 
        new_label = self.label[selected_pts_mask]
        # new_tmp_radii = self.tmp_radii[selected_pts_mask]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_semantic_feature, new_label)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii, opacity_threshold=0.2,only_prune=False):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        # self.tmp_radii = radii
        if not only_prune:
            self.densify_and_clone(grads, max_grad, extent)
            self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size  # self.tmp_radii
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            print(f"Pruning {big_points_vs.sum()} large vs points out of {self.get_xyz.shape[0]} total points.")
            print(f"Pruning {big_points_ws.sum()} large ws points out of {self.get_xyz.shape[0]} total points.")
            # small_points_ws = self.get_scaling.max(dim=1).values < 0.01
            # prune_mask = torch.logical_or(prune_mask, small_points_ws)
            # print(f"Pruning {small_points_ws.sum()} small points out of {self.get_xyz.shape[0]} total points.")

        self.prune_points(prune_mask)
        # self.max_radii2D = self.tmp_radii
        # self.tmp_radii = None
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        grad = viewspace_point_tensor.grad
        if grad.dim() == 3:          # gsplat means2d: [C, N, 2] -> [N, 2]
            grad = grad[0]
        self.xyz_gradient_accum[update_filter] += torch.norm(grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


    def get_smallest_axis(self, return_idx=False):  
        """Returns the smallest axis of the Gaussians.

        Args:
            return_idx (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        rotation_matrices = quaternion_to_matrix(self._rotation)
        smallest_axis_idx = self._scaling.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
        smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
        if return_idx:
            return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
        return smallest_axis.squeeze(dim=2)

    def get_cls_mask(self, label_threshold=0.5, mode='dual'): #mode: geo, se, both 
        """
        Get the mask for structural primitives based on the label threshold.
        """
        # p_geo & p_se
        if self.strprs is None: 
            p_geo = self.label_activation(self.label)
            branch_mask_geo = p_geo > label_threshold
            p_se = self.get_semantic_prior(self.get_semantic_feature)
        else:
            p_geo = self.label_activation(self.strprs.label)
            branch_mask_geo = p_geo > label_threshold
            p_se = self.get_semantic_prior(self.strprs.get_semantic_feature)
        if mode == 'geo':
            p = p_geo
        elif mode == 'se':
            p = p_se
        elif mode == 'dual':
            p = (p_geo + p_se) / 2.
        else:
            raise NotImplementedError
        branch_mask = p > label_threshold
        branch_mask = branch_mask.squeeze()
        leaf_mask = ~branch_mask
        return branch_mask, leaf_mask, p.squeeze()

    def compute_gaussian_overlap_with_neighbors(
        self, 
        neighbor_idx,
        use_gaussian_center_only=False,
        n_samples_to_compute_overlap=4,
        weight_by_normal_angle=False,
        propagate_gradient_to_points_only=False,
        only_branch=False
        ):
        if only_branch:
            branch_mask = torch.sigmoid(self.label) < 0.5
        else:
            branch_mask = torch.ones(self._xyz.shape[0], device=self.device, dtype=bool)
        # This is used to skip the first neighbor, which is the point itself
        neighbor_start_idx = 1
        
        # Get sampled points
        point_idx = neighbor_idx[:, 0][branch_mask]  # (n_points, )
        n_points = len(point_idx)
        
        # Decide whether we want to propagate the gradient to the points only, or to the points and the covariance parameters
        if propagate_gradient_to_points_only:
            scaling = self._scaling[branch_mask].detach()
            quaternions = self._rotation[branch_mask].detach()
        else:
            scaling = self._scaling[branch_mask]
            quaternions = self._rotation[branch_mask]
        
        # Samples points in the corresponding gaussians
        if use_gaussian_center_only:
            n_samples_to_compute_overlap = 1
            gaussian_samples = self._xyz[point_idx].unsqueeze(1) + 0.  # (n_points, n_samples_to_compute_overlap, 3)
        else:
            gaussian_samples = self._xyz[point_idx].unsqueeze(1) + quaternion_apply(
                quaternions[point_idx].unsqueeze(1), 
                scaling[point_idx].unsqueeze(1) * torch.randn(
                    n_points, n_samples_to_compute_overlap, 3, 
                    device=self.device)
                )  # (n_points, n_samples_to_compute_overlap, 3)
        
        # >>> We will now compute the gaussian weight of all samples, for each neighbor gaussian.
        # We start by computing the shift between the samples and the neighbor gaussian centers.
        neighbor_center_to_samples = gaussian_samples.unsqueeze(1) - self._xyz[neighbor_idx[:, neighbor_start_idx:]].unsqueeze(2)  # (n_points, n_neighbors-1, n_samples_to_compute_overlap, 3)
        
        # We compute the inverse of the scaling of the neighbor gaussians. 
        # For 2D gaussians, we implictly project the samples on the plane of each gaussian; 
        # We do so by setting the inverse of the scaling of the gaussian to 0 in the direction of the gaussian normal (i.e. 0-axis).
        inverse_scales = 1. / scaling[neighbor_idx[:, neighbor_start_idx:]].unsqueeze(2)  # (n_points, n_neighbors-1, 1, 3)
        
        # We compute the "gaussian distance" of all samples to the neighbor gaussians, i.e. the norm of the unrotated shift,
        # weighted by the inverse of the scaling of the neighbor gaussians.
        gaussian_distances = inverse_scales * quaternion_apply(
            quaternion_invert(quaternions[neighbor_idx[:, neighbor_start_idx:]]).unsqueeze(2), 
            neighbor_center_to_samples
            )  # (n_points, n_neighbors-1, n_samples_to_compute_overlap, 3)
        
        # Now we can compute the gaussian weights of all samples, for each neighbor gaussian.
        # We then sum them to get the gaussian overlap of each neighbor gaussian.
        gaussian_weights = torch.exp(-1./2. * (gaussian_distances ** 2).sum(dim=-1))  # (n_points, n_neighbors-1, n_samples_to_compute_overlap)
        gaussian_overlaps = gaussian_weights.mean(dim=-1)  # (n_points, n_neighbors-1)
        
        # If needed, we weight the gaussian overlaps by the angle between the normal of the neighbor gaussian and the normal of the point gaussian
        if weight_by_normal_angle:
            normals = self.get_normals()[neighbor_idx]  # (n_points, n_neighbors, 3)
            weights = (normals[:, 1:] * normals[:, 0:1]).sum(dim=-1).abs()  # (n_points, n_neighbors-1)
            gaussian_overlaps = gaussian_overlaps * weights
            
        return gaussian_overlaps
    
    def compute_gaussian_alignment_with_neighbors(
        self,
        neighbor_idx,
        weight_by_normal_angle=False,
        propagate_gradient_to_points_only=False,
        std_factor = 1.,
        ):
        
        # This is used to skip the first neighbor, which is the point itself
        neighbor_start_idx = 1
        
        # Get sampled points
        point_idx = neighbor_idx[:,]  # (n_points, )
        n_points = len(point_idx)
        
        # Decide whether we want to propagate the gradient to the points only, or to the points and the covariance parameters
        if propagate_gradient_to_points_only:
            scaling = self._scaling.detach()
            quaternions = self._rotation.detach()
        else:
            scaling = self._scaling
            quaternions = self._rotation
        
        # We compute scaling, inverse quaternions and centers for all gaussians and their neighbors
        all_scaling = scaling[neighbor_idx]
        all_invert_quaternions = quaternion_invert(quaternions)[neighbor_idx]
        all_centers = self._xyz[neighbor_idx]
        
        # We compute direction vectors between the gaussians and their neighbors
        neighbor_shifts = all_centers[:, neighbor_start_idx:] - all_centers[:, :neighbor_start_idx]
        neighbor_distances = neighbor_shifts.norm(dim=-1).clamp(min=1e-8)
        neighbor_directions = neighbor_shifts / neighbor_distances.unsqueeze(-1)
        
        # We compute the standard deviations of the gaussians in the direction of their neighbors,
        # and reciprocally in the direction of the gaussians.
        standard_deviations_gaussians = (
            all_scaling[:, 0:neighbor_start_idx]
            * quaternion_apply(all_invert_quaternions[:, 0:neighbor_start_idx], 
                               neighbor_directions)
            ).norm(dim=-1)
        
        standard_deviations_neighbors = (
            all_scaling[:, neighbor_start_idx:]
            * quaternion_apply(all_invert_quaternions[:, neighbor_start_idx:], 
                               neighbor_directions)
            ).norm(dim=-1)
        
        # The distance between the gaussians and their neighbors should be the sum of their standard deviations (up to a factor)
        stabilized_distance = (standard_deviations_gaussians + standard_deviations_neighbors) * std_factor
        gaussian_alignment = (neighbor_distances / stabilized_distance.clamp(min=1e-8) - 1.).abs()
        
        # If needed, we weight the gaussian alignments by the angle between the normal of the neighbor gaussian and the normal of the point gaussian
        if weight_by_normal_angle:
            normals = self.get_normals()[neighbor_idx]  # (n_points, n_neighbors, 3)
            weights = (normals[:, 1:] * normals[:, 0:1]).sum(dim=-1).abs()  # (n_points, n_neighbors-1)
            gaussian_alignment = gaussian_alignment * weights
            
        return gaussian_alignment

    def get_neighbors_of_random_points(self, num_samples):
        if num_samples >= 0:
            sampleidx = torch.randperm(len(self._xyz), device=self.device)[:num_samples]        
            return self.knn_idx[sampleidx]
        else:
            return self.knn_idx

    def merge_gaussians(self, gaussian_overlaps, nn_index,threshold=0.5):
        """
        Merge Gaussians in a GaussianModel based on overlap.

        Args:
            gaussian_model: GaussianModel instance
            gaussian_overlaps (torch.Tensor): (N, N) overlap matrix.
            threshold (float): Overlap threshold for merging.
        """
        clusters = self.find_merge_groups(gaussian_overlaps, threshold)

        new_xyz = []
        new_scaling = []
        new_rotation = []
        new_opacity = []
        new_features_dc = []
        new_features_rest = []
        new_max_radii2D = []
        new_tmp_radii = []
        new_denom = []
        new_xyz_gradient_accum = []
        new_exposure = []

        for cluster in clusters:
            if len(cluster) == 1:
                # If only one Gaussian, keep it unchanged
                idx = cluster[0]
                new_xyz.append(self._xyz[idx])
                new_scaling.append(self._scaling[idx])
                new_rotation.append(self._rotation[idx])
                new_opacity.append(self._opacity[idx])
                new_features_dc.append(self._features_dc[idx])
                new_features_rest.append(self._features_rest[idx])
                new_max_radii2D.append(self.max_radii2D[idx])
                new_tmp_radii.append(self.tmp_radii[idx])
                new_denom.append(self.denom[idx])
                new_xyz_gradient_accum.append(self.xyz_gradient_accum[idx])
                new_exposure.append(self._exposure[idx])
            else:
                # Merge Gaussians in the cluster
                indices = torch.tensor(cluster, device=self.device)
                merged_xyz = torch.mean(self._xyz[indices], dim=0)
                merged_scaling = torch.mean(self._scaling[indices], dim=0)
                merged_rotation = torch.mean(self._rotation[indices], dim=0)  # Simple average
                merged_opacity = torch.mean(self._opacity[indices])  # Simple average

                new_xyz.append(merged_xyz)
                new_scaling.append(merged_scaling)
                new_rotation.append(merged_rotation)
                new_opacity.append(merged_opacity)
                new_features_dc.append(self._features_dc[indices[0]])  # Use the first Gaussian's features
                new_features_rest.append(self._features_rest[indices[0]])  # Use the first Gaussian's features
                new_max_radii2D.append(self.max_radii2D[indices].max())
                new_tmp_radii.append(self.tmp_radii[indices].max())
                new_denom.append(self.denom[indices].sum())
                new_xyz_gradient_accum.append(self.xyz_gradient_accum[indices].sum())
                new_exposure.append(self._exposure[indices[0]])  # Use the first Gaussian's exposure
                
        new_opacity_fixed = [op.unsqueeze(0) if op.dim() == 0 else op for op in new_opacity]
        new_xyz = torch.stack(new_xyz)
        new_scaling = torch.stack(new_scaling)
        new_rotation = torch.stack(new_rotation)
        new_opacity_fixed = torch.stack(new_opacity_fixed)
        new_features_dc = torch.stack(new_features_dc)
        new_features_rest = torch.stack(new_features_rest)
        new_max_radii2D = torch.tensor(new_max_radii2D, device=self.device)
        new_tmp_radii = torch.tensor(new_tmp_radii, device=self.device)
        new_denom = torch.tensor(new_denom, device=self.device)
        new_xyz_gradient_accum = torch.tensor(new_xyz_gradient_accum, device=self.device)
        new_exposure = torch.stack(new_exposure)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity_fixed, new_scaling, new_rotation, new_tmp_radii)
        
        print(f"Merged Gaussians. New count: {len(new_xyz)}")

    def find_merge_groups(self,gaussian_overlaps, threshold=0.5):
        """
        Find connected components in the Gaussian overlap graph.
        
        Args:
            gaussian_overlaps (torch.Tensor): (N, N) overlap matrix.
            threshold (float): Overlap threshold for merging.

        Returns:
            List[List[int]]: List of clusters, where each cluster is a list of indices.
        """
        N = gaussian_overlaps.shape[0]
        adjacency_matrix = (gaussian_overlaps > threshold).float()
        
        # Union-Find to find connected components
        parent = list(range(N))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]

        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                parent[root_y] = root_x  # Merge groups

        # Build the merge groups
        for i in range(N):
            for j in range(i + 1, N):  # Only upper triangle to avoid duplicates
                if adjacency_matrix[i, j] > 0:
                    union(i, j)

        # Group Gaussians by connected components
        clusters = {}
        for i in range(N):
            root = find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i)

        return list(clusters.values())

    def get_probs_from_appgas(self, tau: float = 0.07):
        """
        Semantic-only StrPr classification from bound AppGa features.

        Returns:
            p_branch: (S,) tensor in [0,1], where S = #StrPr.
                    1 = branch, 0 = leaf.
        Assumptions:
        - self.appgas.get_semantic_feature() -> (M, D) or (M, 1, D)
        - self.nn_stpr_appgas: (M, 1) or (M,) mapping each AppGa -> StrPr index
        - self.text_feats: (2, D) ordered as [leaf, branch]
        """
        # ---- 1) Fetch & normalize features ----
        feat = self.appgas.get_semantic_feature         # (M, D) or (M,1,D)
        if feat.ndim == 3 and feat.size(1) == 1:
            feat = feat.squeeze(1)                              # -> (M, D)
        feat = F.normalize(feat, dim=-1)                        # AppGa features

        text = F.normalize(self.text_feats, dim=-1)             # [2, D] (leaf, branch)

        nn = self.nn_stpr_appgas
        if nn.ndim == 2 and nn.size(1) == 1:
            nn = nn.squeeze(1)                                  # -> (M,)
        nn = nn.long()

        # Number of StrPrs
        S = int(getattr(self, "num_strpr", (nn.max().item() + 1)))

        # ---- 2) Pool AppGa features -> StrPr features (mean per StrPr) ----
        M, D = feat.shape
        out = torch.zeros(S, D, device=feat.device, dtype=feat.dtype)
        cnt = torch.zeros(S, device=feat.device, dtype=feat.dtype)

        # scatter add features and counts
        out.index_add_(0, nn, feat)
        cnt.index_add_(0, nn, torch.ones(M, device=feat.device, dtype=feat.dtype))

        # avoid div-by-zero; for empty StrPrs keep zero vector for now
        cnt_safe = torch.clamp(cnt, min=1.0)
        strpr_feat = out / cnt_safe.unsqueeze(-1)

        # For StrPrs with no AppGas (cnt==0): set prob=0.5 later by making cosine=0
        # L2-normalize pooled features where count>0; keep zeros for empty ones
        mask_nonempty = (cnt > 0)
        if mask_nonempty.any():
            strpr_feat[mask_nonempty] = F.normalize(strpr_feat[mask_nonempty], dim=-1)

        # ---- 3) Cosine scores -> softmax probabilities (semantic prior) ----
        # scores[:,0]=leaf, scores[:,1]=branch
        scores = strpr_feat @ text.T                            # (S, 2)
        logits = scores / tau
        probs = F.softmax(logits, dim=-1)                       # (S, 2)

        p_branch = probs[:, 1]                                  # (S,)

        # For empty StrPrs (no semantic evidence), force 0.5
        if (~mask_nonempty).any():
            p_branch = p_branch.clone()
            p_branch[~mask_nonempty] = 0.5
        self.strprs.rewrite_semantic_feature(strpr_feat.detach().unsqueeze(1))
        return p_branch

    def compute_binding_loss(self, compute_for_appgas=False, prior='geo', detach_label=False, disk_tol=3.0, normalize=True): # 'geo' or 'semantic'
        disk_params, cylinder_params, mesh_cylinder_list, mesh_disk_list = self.build_primitive_surface(compute_all=True, create_mesh=False)
        if compute_for_appgas:
            appgas_xyz = self.appgas._xyz
        else:
            appgas_xyz = self.appgas._xyz.detach()
        self.update_nn_between_appgas_and_stprs()
        nn = self.nn_stpr_appgas.squeeze(1)
        # Widen the disk in-plane extent for the binding distance only (disk_tol): a leaf's
        # planar spread reaches ~2-3 std, but the disk a,b are ~1 std, so half the leaf points
        # fall outside the ellipse and get penalised -> they wrongly prefer the (thin) cylinder.
        # Mesh disks keep their true size; this only affects the branch/leaf label signal.
        disk_bind = {**disk_params, 'a': disk_params['a'] * disk_tol, 'b': disk_params['b'] * disk_tol}
        dis_cylinder = gs_to_cylinder_distance(appgas_xyz, nn, cylinder_params)
        dis_disk = gs_to_disk_distance(appgas_xyz, nn, disk_bind)
        label_for_prob = self.strprs.label.detach() if detach_label else self.strprs.label
        if prior == 'geo':
            prob = self.label_activation(label_for_prob)[nn].squeeze()
        else:
            prob = self.get_probs_from_appgas()[nn].squeeze()
        if normalize:
            # Per-StrPr NORMALISED binding: keep the paper's p*d_cyl + (1-p)*d_disk form (so the
            # geometric binding still drives the branch/leaf label and AppGS gradients still flow
            # to StrPr), but divide each StrPr's cost by (D_cyl+D_disk) so the label gradient is
            # (D_cyl-D_disk)/(D_cyl+D_disk) in [-1,1] -- O(1), no longer swamped by the colour loss.
            S = self.strprs.get_xyz.shape[0]
            dev = appgas_xyz.device
            D_cyl = torch.zeros(S, device=dev); D_disk = torch.zeros(S, device=dev); cnt = torch.zeros(S, device=dev)
            D_cyl.index_add_(0, nn, dis_cylinder)
            D_disk.index_add_(0, nn, dis_disk)
            cnt.index_add_(0, nn, torch.ones_like(dis_cylinder))
            has = cnt > 0
            D_cyl = D_cyl / cnt.clamp_min(1.0); D_disk = D_disk / cnt.clamp_min(1.0)
            p_s = self.label_activation(label_for_prob).squeeze()
            denom = (D_cyl + D_disk).clamp_min(1e-6)
            per = (p_s * D_cyl + (1.0 - p_s) * D_disk) / denom
            loss_bind = per[has].mean() if has.any() else per.sum() * 0.0
            return loss_bind, D_cyl[has].mean(), D_disk[has].mean()
        dis_cylinder = dis_cylinder * prob
        dis_disk = dis_disk * (1-prob)
        loss_bind = dis_cylinder.mean() + dis_disk.mean()
        return loss_bind, dis_cylinder.mean(), dis_disk.mean()
    
    def build_primitive_surface(self, iteration=0,label_threshold=0.5, compute_all=False, branch_mask=None,vis_bindloss=False,create_mesh=False):
        if branch_mask is None:
            if compute_all:
                branch_mask = torch.ones_like(self.strprs.label, dtype=torch.bool).squeeze()
                leaf_mask = branch_mask
            else:
                branch_mask, leaf_mask, _ = self.get_cls_mask(label_threshold=label_threshold)
        else:
            leaf_mask = ~branch_mask
        # cylinder part
        branch_pos = self.strprs.get_xyz[branch_mask]
        branch_rot = build_rotation(self.strprs.get_rotation[branch_mask])
        # branch_rot_cylinder = self.strprs._rotation_cylinder[branch_mask]
        branch_scale = self.strprs.get_scaling[branch_mask]
        cylinder_params, mesh_cylinder_all, mesh_cylinder_list = strpr_to_cylinder(branch_pos,
                                                                  branch_scale,
                                                                  branch_rot,
                                                                  iteration,
                                                                  create_mesh=create_mesh,
                                                                  save_path ='./cylinder.ply',
                                                                  resolution=32)
        # disk part
        leaf_pos = self.strprs.get_xyz[leaf_mask]
        leaf_rot = build_rotation(self.strprs.get_rotation[leaf_mask])
        leaf_scale = self.strprs.get_scaling[leaf_mask]
        disk_params, mesh_disk_all, mesh_disk_list = strpr_to_disk(leaf_pos,
                                                                      leaf_scale,
                                                                      leaf_rot,
                                                                      iteration,
                                                                      create_mesh=create_mesh,
                                                                      save_path='./disk.ply',
                                                                      resolution=32)
        return disk_params, cylinder_params, mesh_cylinder_list, mesh_disk_list
        
    def build_branch_graph(self, mode="gsplant"):
        branch_mask, leaf_mask, _ = self.get_cls_mask(label_threshold=0.5, mode='geo')
        scale_mask = self.strprs.get_scaling.max(dim=1).values > 0.001
        if mode == "gsplant":
            branch_mask = torch.logical_and(branch_mask, scale_mask)
        elif mode=="branch":
            branch_mask = scale_mask
        center = self.strprs.get_xyz[branch_mask]
        n_strpr = center.shape[0]
        if n_strpr == 0:  # no branch StrPr -> empty graph (avoid arange/stack errors)
            empty = torch.zeros((0, 3), device=self.strprs.get_xyz.device)
            return torch.zeros((0, 2), dtype=torch.long, device=empty.device), empty, empty
        rotation = build_rotation(self.strprs.get_rotation[branch_mask])
        # Cylinder axis = the gaussian's MAJOR (largest-scale) axis, not a fixed column. After
        # optimisation the long axis lands on whichever local axis the L1 elongated (argmax of the
        # 3 scales), which is column 1 or 2 for most branch StrPr -> using column 0 pointed the
        # skeleton ~the wrong way ("x/y swapped"). Pick the argmax-scale column per StrPr.
        b_scaling = self.strprs.get_scaling[branch_mask]
        major = b_scaling.argmax(dim=1)                       # (n,) which local axis is longest
        idx = torch.arange(n_strpr, device=center.device)
        u = rotation[idx, :, major]                          # (n,3) major-axis direction
        h = b_scaling[idx, major] * 1.5
        top, bottom = center + h[:, None] * u, center - h[:, None] * u
        p = torch.stack((top, bottom), dim=1).reshape(-1,3)  # (N,2,3)
        strpr_pairs = torch.stack([
        torch.arange(0, 2*n_strpr, 2, device=p.device),
        torch.arange(1, 2*n_strpr, 2, device=p.device)
    ], dim=1)
        mst_edges = build_mst_from_endpoints(p,strpr_pairs)
        # dense appgas branch points 
        self.update_nn_between_appgas_and_stprs()
        branch_appgas_mask = branch_mask[self.nn_stpr_appgas.squeeze(1)]
        appgas_branch_points = self.appgas.get_xyz[branch_appgas_mask]
        return mst_edges, p, appgas_branch_points

    def densify_branches(self, ratio_thr=1.5, max_strpr_num=6000):
        """Structure-driven, BRANCH-ONLY densification: split a branch StrPr along its major axis
        when the AppGS it binds spill past its cylinder (along-axis extent > ratio_thr * length).

        This is the right densify signal for a skeleton: NOT photometric gradient (that chases leaf
        texture and wrecks the branch binding), but the binding geometry itself -- a cylinder that
        under-covers the branch segment it is responsible for is split into two shorter cylinders
        placed along that segment. Leaves/AppGS are untouched. CD (bound-AppGS Chamfer) is ~invariant
        (the AppGS set does not change); what improves is the fineness/faithfulness of the skeleton.
        Returns the number of StrPr split.
        """
        st = self.strprs
        if st.get_xyz.shape[0] >= max_strpr_num:
            return 0
        p_geo = st.label_activation(st.label).squeeze(-1)
        branch_mask = (p_geo > 0.5) & (st.get_scaling.max(dim=1).values > 0.001)
        if int(branch_mask.sum()) < 1:
            return 0
        self.update_nn_between_appgas_and_stprs()
        nn = self.nn_stpr_appgas.squeeze(1)
        axyz = self.appgas.get_xyz.detach()
        R = build_rotation(st.get_rotation)
        scl = st.get_scaling
        amax = scl.argmax(dim=1)
        bidx = torch.where(branch_mask)[0]
        sel, c1s, c2s, scales_raw = [], [], [], []
        for i in bidx:
            m = nn == i
            if int(m.sum()) < 3:
                continue
            u = R[i, :, amax[i]]
            proj = ((axyz[m] - st.get_xyz[i]) * u).sum(1)
            pmin, pmax = proj.min(), proj.max()
            extent = (pmax - pmin)
            length = 2.0 * scl[i, amax[i]] * 1.5                  # cylinder length (build_branch_graph)
            if extent / length.clamp_min(1e-6) <= ratio_thr:
                continue
            mid = 0.5 * (pmin + pmax)
            off = 0.25 * extent                                   # two children at the quarter points
            ns = st._scaling[i].clone()                           # keep perp axes, shrink major
            ns[amax[i]] = st.scaling_inverse_activation((extent / 2.0 / 3.0).clamp_min(1e-4))
            sel.append(int(i))
            c1s.append(st.get_xyz[i] + u * (mid - off))
            c2s.append(st.get_xyz[i] + u * (mid + off))
            scales_raw.append(ns)
        if not sel:
            return 0
        sel_t = torch.tensor(sel, device=st.get_xyz.device)
        # block order: [all child-1, all child-2] to match the repeat(2,...) below
        new_xyz = torch.cat([torch.stack(c1s, 0), torch.stack(c2s, 0)], dim=0)   # [2k,3]
        new_scaling = torch.stack(scales_raw, dim=0).repeat(2, 1)  # children share parent's perp scales
        new_rotation = st._rotation[sel_t].repeat(2, 1)
        new_features_dc = st._features_dc[sel_t].repeat(2, 1, 1)
        new_features_rest = st._features_rest[sel_t].repeat(2, 1, 1)
        new_opacity = st._opacity[sel_t].repeat(2, 1)
        new_semantic = st._semantic_feature[sel_t].repeat(2, 1, 1)
        new_label = st.label[sel_t].repeat(2, 1)
        st.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity,
                                 new_scaling, new_rotation, new_semantic, new_label)
        prune = torch.zeros(st.get_xyz.shape[0], dtype=torch.bool, device=st.get_xyz.device)
        prune[sel_t] = True
        st.prune_points(prune)
        return len(sel)

    def compute_branch_axis_alignment(self, k=6):
        """Encourage each BRANCH StrPr to be elongated ALONG the local branch direction, so the
        cylinders line up head-to-tail instead of crossing the branch (the major axis was ~50 deg
        off the branch tangent, with nothing aligning it).

        tangent_i = principal direction of the k nearest branch-StrPr centres (local PCA of the
        skeleton). The loss pushes the gaussian's variance onto that tangent:
            loss = 1 - std_along_tangent / std_total
        Minimising it rotates/stretches each branch StrPr so its long axis follows the branch.
        Because neighbouring StrPr share (nearly) the same local tangent, their major axes also
        become mutually consistent -- the "align neighbouring major axes" the user asked for.
        """
        p_geo = self.label_activation(self.strprs.label).squeeze(-1)        # geo branch prob
        branch_mask = (p_geo > 0.5) & (self.strprs.get_scaling.max(dim=1).values > 0.001)
        n = int(branch_mask.sum())
        dev = self.strprs.get_xyz.device
        if n < 4:
            return torch.tensor(0.0, device=dev)
        xyz = self.strprs.get_xyz[branch_mask]
        R = build_rotation(self.strprs.get_rotation[branch_mask])          # [n,3,3], cols=local axes
        # Major-axis DIRECTION = rotation column of the largest scale. The axis SELECTION uses
        # detached scales (argmax is non-differentiable and we must not let the loss shrink the
        # radius); the column vector itself stays differentiable so the gradient only ROTATES the
        # gaussian. A pure angular loss (1-|cos|) aligns the existing long axis to the branch
        # tangent WITHOUT touching the scales -> no needle collapse (the earlier variance-ratio
        # form drove the radius to ~0, aniso 1400+).
        amax = self.strprs.get_scaling[branch_mask].detach().argmax(dim=1)
        u = R[torch.arange(n, device=xyz.device), :, amax]                # [n,3] major-axis dir
        kk = min(k, n)
        with torch.no_grad():
            knn = torch.cdist(xyz, xyz).topk(kk, largest=False).indices    # [n,kk] (includes self)
            nb = xyz[knn]
            nb = nb - nb.mean(dim=1, keepdim=True)
            cov = torch.einsum('nki,nkj->nij', nb, nb) / kk               # [n,3,3]
            tangent = torch.linalg.eigh(cov).eigenvectors[:, :, -1]       # principal dir [n,3]
            tangent = F.normalize(tangent, dim=1)
        cos = (F.normalize(u, dim=1) * tangent).sum(dim=1).abs()          # |cos angle|, sign-free
        return (1.0 - cos).mean()

    def geometric_loss(self):
        source = self.appgas._xyz.detach()
        target = self.strprs.sample_points_in_gaussians(num_samples=10000)
        # Compute the distance between the source and target points
        loss_geo = chamfer_distance(source.unsqueeze(0), target[0].unsqueeze(0))[0]
        return loss_geo.mean()

    @torch.no_grad()
    def compute_plant_mask(self, cameras, thresh=0.5, min_views=3, mask_value=0.5):
        """Per-Gaussian plant/background mask by projecting into the 2D plant masks.

        A Gaussian is 'plant' if, among the views where it is inside the frustum, it
        lands inside the plant mask in more than ``thresh`` of them (and is seen in at
        least ``min_views`` views). Removes the pot/background that dominates the cloud
        (~86% here) so colour can cleanly separate branch (brown) from leaf (green).
        """
        xyz = self.get_xyz
        N = xyz.shape[0]
        homo = torch.cat([xyz, torch.ones(N, 1, device=xyz.device)], dim=1)  # [N,4]
        inside = torch.zeros(N, device=xyz.device)
        visible = torch.zeros(N, device=xyz.device)
        for cam in cameras:
            if cam.mask is None:
                continue
            W2C = cam.world_view_transform.transpose(0, 1)
            pc = (homo @ W2C.T)[:, :3]
            z = pc[:, 2]
            W, H = int(cam.image_width), int(cam.image_height)
            fx = W / (2.0 * math.tan(cam.FoVx * 0.5))
            fy = H / (2.0 * math.tan(cam.FoVy * 0.5))
            u = fx * pc[:, 0] / z + W * 0.5
            v = fy * pc[:, 1] / z + H * 0.5
            inb = (z > 0.01) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            visible += inb.float()
            mask = cam.mask.to(xyz.device)
            ui = u.clamp(0, W - 1).long()
            vi = v.clamp(0, H - 1).long()
            hit = torch.zeros(N, device=xyz.device)
            hit[inb] = (mask[vi[inb], ui[inb]] > mask_value).float()
            inside += hit
        ratio = inside / visible.clamp_min(1)
        return (ratio > thresh) & (visible >= min_views)

    def apply_mask(self, mask):
        """Subset all per-Gaussian tensors in-place (use before training_setup)."""
        for attr in ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation',
                     '_opacity', '_semantic_feature', 'label', 'max_radii2D']:
            t = getattr(self, attr, None)
            if t is not None and t.shape[0] == mask.shape[0]:
                sub = t[mask]
                setattr(self, attr, nn.Parameter(sub.detach().requires_grad_(True))
                        if isinstance(t, nn.Parameter) else sub)
        print(f"[GSplantModel] Background removed: kept {int(mask.sum())}/{mask.shape[0]} plant Gaussians.")

    def build_strpr_from_gs(
        self,
        cluster_method: Literal["kmeans", "dbscan"] = "kmeans",
        vis_cluster=False,
        max_strpr_num: int= 1000,
        filter_small_cluster=True,
        label_init: str = "joint",   # "joint" (color+geometry+semantic) | "anisotropy" | "semantic" | "color"
        w_col: float = 2.0,
        w_geo: float = 1.0,
        w_sem: float = 0.5,
        branch_frac: float = 0.08,
        cluster_size: int = 100,      # avg points per StrPr cluster (smaller -> finer/more StrPr)
    ):
        """
        Structural Primitives (StPrs) creation from optimized 2D Gaussian splats.

        Args:
            num_clusters: Number of clusters.
            cluster_method: "kmeans" or "dbscan".
            denoise: Removes isolated Gaussians.
            anisotropy_thresh: Threshold for branch-leaf anisotropy.
        """

        xyz = self._xyz.detach()                          # (N,3) tensor on GPU           # (N,4)
        feature_dc = self._features_dc.detach()
        normals = self.get_normals.detach() # (N,1,3)
        feature_rest = self._features_rest.detach()       # (N, sh_coeff,3)
        strpr_positions = []
        strpr_scales = []
        strpr_rotations = []
        strpr_labels = []
        strpr_features_dc = []
        strpr_features_rest = []
        strpr_semantic_feature = []
        strpr_cylinder_rotations = []
        strpr_disk_rotations = []
        strpr_anisotropy = []
        # Clustering (GPU-KMeans via FAISS)
        if cluster_method == "kmeans":
            # feature = torch.cat([xyz, normals], dim=1)
            num_clusters = max(2, xyz.shape[0] // cluster_size)
            label_path = os.path.join(self.save_path, f"kmeans_labels_{num_clusters}.npy")
            if os.path.exists(label_path):
                labels = np.load(label_path)
                print(f"[GSplantModel] Loaded existing labels from {label_path}.")
            else:
                feature = xyz.cpu().numpy().astype(np.float32)  # Convert to numpy for FAISS
                kmeans = faiss.Kmeans(d=3, k=num_clusters, niter=50, nredo=3,gpu=True)
                kmeans.train(feature)
                labels = kmeans.index.search(feature, 1)[1].flatten()          
                np.save(label_path, labels)
        elif cluster_method == "dbscan":        
            """
            parameters:
            syn_plant2: eps: 0.01, min_samples: 3
            plant1: eps:0.03, min_samples: 5
            plant0: eps:0.05, min_samples:3
            """
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=0.03, min_samples=5).fit(xyz.cpu().numpy())
            labels = clustering.labels_
        else:
            raise ValueError(f"Unsupported clustering method: {cluster_method}")
        if vis_cluster:
            save_labeled_points(xyz.cpu().numpy(), labels, self.save_path)


        for cluster_id in range(len(set(labels))):
            cluster_mask = [labels== cluster_id]
            cluster_points = xyz[cluster_mask]
            if len(cluster_points) > 5:
                mean,stpr_rot,stpr_scale, anisotropy,rot_matrix_cylinder,rot_matrix_disk = estimate_gs_para_from_cluster(cluster_points)
                strpr_positions.append(mean)
                strpr_scales.append(stpr_scale)
                strpr_rotations.append(stpr_rot)
                strpr_anisotropy.append(torch.abs(anisotropy))
                strpr_features_dc.append(feature_dc[cluster_mask].mean(dim=0))
                strpr_features_rest.append(feature_rest[cluster_mask].mean(dim=0))
                strpr_cylinder_rotations.append(rot_matrix_cylinder)
                strpr_disk_rotations.append(rot_matrix_disk)
                strpr_semantic_feature.append(self._semantic_feature[cluster_mask].mean(dim=0))
                strpr_labels.append(self.get_semantic_prior(self._semantic_feature[cluster_mask]).mean())
        # Convert lists to tensors (GPU-friendly)
        strpr_positions = torch.stack(strpr_positions, dim=0).to(self.device)
        strpr_scales = self.scaling_inverse_activation(torch.stack(strpr_scales, dim=0)).to(self.device)
        # check nan in strpr_scales
        if torch.isnan(strpr_scales).any():
            strpr_scales[torch.isnan(strpr_scales)] = 0.01
        strpr_rotations = torch.stack(strpr_rotations, dim=0).to(self.device)
        strpr_features_dc = torch.stack(strpr_features_dc, dim=0).to(self.device)
        strpr_features_rest = torch.stack(strpr_features_rest, dim=0).to(self.device)
        strpr_disk_rotations = torch.stack(strpr_disk_rotations, dim=0).to(self.device)
        strpr_anisotropy = torch.stack(strpr_anisotropy, dim=0).to(self.device)
        strpr_cylinder_rotations = torch.stack(strpr_cylinder_rotations, dim=0).to(self.device)
        strpr_semantic_feature = torch.stack(strpr_semantic_feature, dim=0).to(self.device)
        strpr_opacities = self.inverse_opacity_activation(0.5 * torch.ones((len(strpr_positions), 1), device=self.device))
        # per-cluster semantic branch prior (mean of get_semantic_prior over the cluster)
        strpr_sem_prior = torch.stack(strpr_labels, dim=0).unsqueeze(1).to(self.device)  # [K,1]

        # --- Branch/leaf label initialisation ---
        # Two cues, both weak on their own (geometry ~0.62 AUC, semantics ~0.66 AUC),
        # complementary together. We standardise each across clusters and combine into a
        # single score, then map to a soft label in (0.4, 0.6) ordered by that score, so
        # the subsequent optimisation can move it freely. No hard threshold / per-scene tuning.
        if label_init == "joint":
            def _z(x):
                x = x.flatten().float()
                return (x - x.mean()) / (x.std() + 1e-6)
            # color is by far the strongest cue for plants (green leaves vs brown branches,
            # AUC ~0.89) — branch-ness == NOT green. Geometry/semantics are weaker complements.
            rgb = 0.2820948 * strpr_features_dc.squeeze(1) + 0.5      # SH DC -> approx base color
            greenness = rgb[:, 1] - 0.5 * (rgb[:, 0] + rgb[:, 2])
            z_col = _z(-greenness)                                    # high => branch (not green)
            z_geo = _z(torch.log(strpr_anisotropy.clamp_min(1e-6)))   # geometry: cluster elongation
            z_sem = _z(strpr_sem_prior)                              # semantics: DINOv3 text prior
            score = w_col * z_col + w_geo * z_geo + w_sem * z_sem
            # Branches are the minority; centre the 0.5 crossing so ~branch_frac start as branch.
            # branch_frac <= 0 -> AUTO-calibrate the threshold by Otsu on the score distribution
            # (the brown/elongated branch mode vs the green/planar leaf mode), no per-scene tuning.
            if branch_frac is None or branch_frac <= 0:
                # AUTO: Otsu valley on the score, capped at auto_cap (branches are a minority;
                # Otsu's balanced split tends to over-estimate when the score is not cleanly
                # bimodal, so cap it). Adaptive pruning then trims residual over-estimates.
                auto_cap = 0.15
                thr = _otsu_threshold(score)
                if (score > thr).float().mean() > auto_cap:
                    thr = torch.quantile(score, 1.0 - auto_cap)
                print(f"[GSplantModel] auto branch_frac = {(score > thr).float().mean().item():.3f} (Otsu, cap {auto_cap})")
            else:
                thr = torch.quantile(score, 1.0 - branch_frac)
            strpr_labels = (0.40 + 0.20 * torch.sigmoid(2.0 * (score - thr))).unsqueeze(1)  # [K,1] in (0.4,0.6)
        elif label_init == "anisotropy":
            strpr_labels = torch.where(
                (strpr_anisotropy > 1.5).unsqueeze(1),
                torch.full_like(strpr_sem_prior, 0.6),
                torch.full_like(strpr_sem_prior, 0.4))
        elif label_init == "semantic":
            strpr_labels = strpr_sem_prior.clamp(0.05, 0.95)
        else:
            raise ValueError(f"Unknown label_init: {label_init}")
        # Create a new GaussianModel for structural primitives
        self.strprs = GaussianModel(sh_degree=self.max_sh_degree, device=self.device, save_path=self.save_path)
        self.strprs._xyz = nn.Parameter(strpr_positions.requires_grad_(True))
        self.strprs._scaling = nn.Parameter(strpr_scales.requires_grad_(True))
        self.strprs._rotation = nn.Parameter(strpr_rotations.requires_grad_(True))
        self.strprs._features_dc = nn.Parameter(strpr_features_dc.requires_grad_(True))
        self.strprs._features_rest = nn.Parameter(strpr_features_rest.requires_grad_(True))
        self.strprs._opacity = nn.Parameter(strpr_opacities.requires_grad_(True))
        self.strprs.max_radii2D = torch.zeros(len(strpr_positions), device=self.device)
        self.strprs._rotation_cylinder = nn.Parameter(strpr_cylinder_rotations.requires_grad_(True))
        self.strprs._rotation_disk = nn.Parameter(strpr_disk_rotations.requires_grad_(True))
        self.strprs.label = nn.Parameter(torch.logit(strpr_labels).requires_grad_(True))
        self.strprs._semantic_feature = nn.Parameter(strpr_semantic_feature.requires_grad_(True))
        self.strprs.is_strprs = True
        self.strprs.text_feats = self.text_feats

        print(f"[GSplantModel] Initialized {len(strpr_positions)} StrPrs.")
        return self.strprs
    
    def build_appgas_from_stprs(self,num_sample=10, use_pretrain=True, branch_mask=None):
        if use_pretrain:
            self.appgas = GaussianModel(sh_degree=self.max_sh_degree, device=self.device, save_path=self.save_path)
            self.appgas._xyz = nn.Parameter(self._xyz.detach().clone().requires_grad_(True))
            self.appgas._scaling = nn.Parameter(self._scaling.detach().clone().requires_grad_(True))
            self.appgas._rotation = nn.Parameter(self._rotation.detach().clone().requires_grad_(True))
            self.appgas._features_dc = nn.Parameter(self._features_dc.detach().clone().requires_grad_(True))
            self.appgas._features_rest = nn.Parameter(self._features_rest.detach().clone().requires_grad_(True))
            self.appgas._opacity = nn.Parameter(self._opacity.detach().clone().requires_grad_(True))
            self.appgas.max_radii2D = torch.zeros(len(self._xyz), device=self.device)
            self.appgas.label = nn.Parameter(self.label.detach().clone().requires_grad_(True))
            self.appgas._semantic_feature = nn.Parameter(self._semantic_feature.detach().clone().requires_grad_(True))
            self.appgas.is_appgas = True
            self.appgas.text_feats = self.text_feats
            self.appgas.strprs = None
        else:
            _,_,mesh_cylinder_list, mesh_disk_list = self.build_primitive_surface(create_mesh=True, branch_mask=branch_mask)
            if branch_mask is None:
                branch_mask = self.label_activation(self.strprs.label) > 0.5
                branch_mask = branch_mask.squeeze()
            leaf_mask = ~branch_mask
            branch_rot = self.strprs.get_rotation[branch_mask]
            branch_scale = self.strprs.get_scaling[branch_mask]
            branch_features_dc = self.strprs._features_dc[branch_mask]
            branch_features_rest = self.strprs._features_rest[branch_mask]
            branch_label = self.strprs.label[branch_mask]
            branch_semantic_feature = self.strprs._semantic_feature[branch_mask]
            leaf_rot = self.strprs.get_rotation[leaf_mask]
            leaf_scale = self.strprs.get_scaling[leaf_mask]
            leaf_features_dc = self.strprs._features_dc[leaf_mask]
            leaf_features_rest = self.strprs._features_rest[leaf_mask]
            leaf_label = self.strprs.label[leaf_mask]
            leaf_semantic_feature = self.strprs._semantic_feature[leaf_mask]
            # build appgas 
            appgas_pos = []
            appgas_rot = []
            appgas_scale = []
            appgas_features_dc = []
            appgas_features_rest = []
            appgas_label = []
            appgas_semantic_feature = []
            for i,cylinder in enumerate(mesh_cylinder_list):
                samples = cylinder.sample_points_uniformly(number_of_points=num_sample)
                appgas_pos.append(torch.tensor(np.asarray(samples.points)).float())
                appgas_rot.append(branch_rot[i].detach().repeat(num_sample, 1, 1))
                appgas_scale.append((branch_scale[i]/10).repeat(num_sample, 1))
                appgas_features_dc.append(branch_features_dc[i].repeat(num_sample, 1, 1))
                appgas_features_rest.append(branch_features_rest[i].repeat(num_sample, 1, 1))
                appgas_label.append(branch_label[i].repeat(num_sample, 1))
                appgas_semantic_feature.append(branch_semantic_feature[i].repeat(num_sample, 1))
            for i,disk in enumerate(mesh_disk_list):
                samples = disk.sample_points_uniformly(number_of_points=num_sample)
                appgas_pos.append(torch.tensor(np.asarray(samples.points)).float())
                appgas_rot.append(leaf_rot[i].detach().repeat(num_sample, 1, 1))
                appgas_scale.append((leaf_scale[i]/10).repeat(num_sample, 1))
                appgas_features_dc.append(leaf_features_dc[i].repeat(num_sample, 1, 1))
                appgas_features_rest.append(leaf_features_rest[i].repeat(num_sample, 1, 1))
                appgas_label.append(leaf_label[i].repeat(num_sample, 1))
                appgas_semantic_feature.append(leaf_semantic_feature[i].repeat(num_sample, 1))
            appgas_pos = torch.vstack(appgas_pos).to(self.device)
            appgas_rot = torch.vstack(appgas_rot).squeeze(1).to(self.device)
            appgas_scale = self.scaling_inverse_activation(torch.vstack(appgas_scale)).to(self.device)
            appgas_features_dc = torch.vstack(appgas_features_dc).to(self.device)
            appgas_features_rest = torch.vstack(appgas_features_rest).to(self.device)
            appgas_opacities = self.inverse_opacity_activation(0.5* torch.ones((appgas_pos.shape[0], 1))).to(self.device)
            appgas_label = torch.vstack(appgas_label).to(self.device)
            appgas_semantic_feature = torch.vstack(appgas_semantic_feature).unsqueeze(1).to(self.device)
            # create appga model
            self.appgas = GaussianModel(sh_degree=self.max_sh_degree, device=self.device, save_path=self.save_path)
            self.appgas._xyz = nn.Parameter(appgas_pos.requires_grad_(True))
            self.appgas._rotation = nn.Parameter(appgas_rot.requires_grad_(True))
            self.appgas._scaling = nn.Parameter(appgas_scale.requires_grad_(True))
            self.appgas._features_dc = nn.Parameter(appgas_features_dc.requires_grad_(True))
            self.appgas._features_rest = nn.Parameter(appgas_features_rest.requires_grad_(True))
            self.appgas._opacity = nn.Parameter(appgas_opacities.requires_grad_(True))
            self.appgas.max_radii2D = torch.zeros(len(appgas_pos)).to(self.device)
            self.appgas.label = nn.Parameter(torch.logit(appgas_label).requires_grad_(True))
            self.appgas._semantic_feature = nn.Parameter(appgas_semantic_feature.requires_grad_(True))
            self.appgas.is_appgas = True
            self.appgas.text_feats = self.text_feats
        return self.appgas

    def update_nn_between_appgas_and_stprs(self, metric='euclidean'):
        app_xyz = self.appgas._xyz.detach()                     # (N_app, 3)
        strpr_xyz = self.strprs._xyz.detach()             # (N_str, 3)
        if metric == "euclidean":
            knn = knn_points(app_xyz[None], strpr_xyz[None], K=self.knn_to_track)
            nn_index = knn.idx[0, :, 0]  # (N_app,)
        if metric == "maha":
            scaling = self.strprs.get_scaling.detach()  # (N_str, 3)
            quats   = self.strprs.get_rotation.detach() # (N_str, 4)

            eps = 1e-4
            sigma = torch.sqrt(torch.clamp(scaling, min=eps))      # (N_str, 3)

            N_app, N_str = app_xyz.shape[0], strpr_xyz.shape[0]
            app_expand  = app_xyz.unsqueeze(1).expand(-1, N_str, -1)   # (N_app, N_str, 3)
            stpr_expand = strpr_xyz.unsqueeze(0).expand(N_app, -1, -1)  # (N_app, N_str, 3)
            shift       = app_expand - stpr_expand                     # (N_app, N_str, 3)
            quats_expand = quats.unsqueeze(0).expand(N_app, -1, -1)    # (N_app, N_str, 4)
            inv_quats    = quaternion_invert(quats_expand)            # (N_app, N_str, 4)
            local_shift  = quaternion_apply(inv_quats, shift)         # (N_app, N_str, 3)

            inv_sigma = 1.0 / sigma.unsqueeze(0)                      # (1, N_str, 3)
            normed    = local_shift * inv_sigma                       # (N_app, N_str, 3)

            maha2 = normed.pow(2).sum(dim=-1)                         # (N_app, N_str)
            maha  = torch.sqrt(maha2 + 1e-8)                          # (N_app, N_str)

            nn_index = maha.argmin(dim=1)   
        self.nn_stpr_appgas = nn_index.unsqueeze(1)  # (N_app, 1)
    
    def sample_points_in_gaussians(self, num_samples, sampling_scale_factor=1., mask=None,
                                   probabilities_proportional_to_opacity=False,
                                   probabilities_proportional_to_volume=True,):
        if mask is None:
            scaling = self._scaling
        else:
            scaling = self._scaling[mask]
        
        if probabilities_proportional_to_volume:
            areas = scaling[..., 0] * scaling[..., 1] * scaling[..., 2]
        else:
            areas = torch.ones_like(scaling[..., 0])
        
        if probabilities_proportional_to_opacity:
            if mask is None:
                areas = areas * self.strengths.view(-1)
            else:
                areas = areas * self.strengths[mask].view(-1)
        areas = areas.abs()
        # cum_probs = areas.cumsum(dim=-1) / areas.sum(dim=-1, keepdim=True)
        cum_probs = areas / areas.sum(dim=-1, keepdim=True)
        
        random_indices = torch.multinomial(cum_probs, num_samples=num_samples, replacement=True)
        if mask is not None:
            valid_indices = torch.arange(self.n_points, device=self.device)[mask]
            random_indices = valid_indices[random_indices]
        
        random_points = self._xyz[random_indices] + quaternion_apply(
            self._rotation[random_indices], 
            sampling_scale_factor * self._scaling[random_indices] * torch.randn_like(self._xyz[random_indices]))
        
        return random_points, random_indices

    def save_stpr_app_correspondence(self, out_name, sphere_rad=0.005):
        xyz_stpr  = self.strprs.get_xyz      # (P,3)
        xyz_app   = self.appgas._xyz                 # (M,3)
        parent_id = self.nn_stpr_appgas.squeeze(1).cpu()             # (M,)

        P = xyz_stpr.shape[0]

        rng = torch.Generator().manual_seed(0)      
        colors_stpr = torch.rand((P, 3), generator=rng)             # (P,3)
        colors_app  = colors_stpr[parent_id]                        # (M,3)

        # build cylinder and disk for stpr
        leaf_mask = torch.sigmoid(self.structure_gs.label) >= 0.5
        branch_mask = ~leaf_mask
        cyl_param,_,cyl_mesh_list = strpr_to_cylinder(
            xyz_stpr[branch_mask], 
            self.structure_gs.get_scaling[branch_mask], 
            self.structure_gs.get_rotation[branch_mask], 
            create_mesh=True)
        
        disk_param, _ ,disk_mesh_list = strpr_to_disk(
            xyz_stpr[leaf_mask], 
            self.structure_gs.get_scaling[leaf_mask], 
            self.structure_gs.get_rotation[leaf_mask], 
            create_mesh=True)
        # corres
        mesh_all = o3d.geometry.TriangleMesh()
        branch_ids = branch_mask.nonzero(as_tuple=False).squeeze(1)
        for idx_local, idx_global in enumerate(branch_ids):
            mesh = cyl_mesh_list[idx_local]
            mesh.paint_uniform_color(colors_stpr[idx_global].numpy())
            mesh_all += mesh
        leaf_ids = leaf_mask.nonzero(as_tuple=False).squeeze(1)
        for idx_local, idx_global in enumerate(leaf_ids):
            mesh = disk_mesh_list[idx_local]
            mesh.paint_uniform_color(colors_stpr[idx_global].numpy())
            mesh_all += mesh
        # app to sphere mesh for vis
        sphere_app_tpl = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_rad,resolution=3)
        sphere_app_tpl.compute_vertex_normals()
        for p, c in zip(xyz_app, colors_app):
            sphere_app = sphere_app_tpl.translate(p.detach().cpu().numpy(), relative=False)
            sphere_app.paint_uniform_color(c.detach().cpu().numpy())
            mesh_all += sphere_app
        mesh_path = f"{out_name}_mesh.ply"
        app_path  = f"{out_name}_app.ply"
        o3d.io.write_triangle_mesh(mesh_path, mesh_all)
        # o3d.io.write_point_cloud(app_path, pcd_app)
        print(f"[save] StPr spheres -> {mesh_path}")
        # print(f"[save] AppG points  -> {app_path}")

    def regularizer_semantic(self):
        feat_appgas = self.appgas.get_semantic_feature.squeeze(1)
        feat_strpr  = self.strprs.get_semantic_feature.squeeze(1)
        self.update_nn_between_appgas_and_stprs()
        nn = self.nn_stpr_appgas.squeeze(1)
        mu_for_appga = feat_strpr[nn]
        L_intra = ((feat_appgas - mu_for_appga).pow(2).mean())
        return L_intra

    def semantic_loss(self, tau: float = 0.07):
        p_geo = self.strprs.label_activation(self.strprs.label).squeeze(1)  # (N,)
        self.update_nn_between_appgas_and_stprs()
        f_app = self.appgas.get_semantic_feature.squeeze(1).detach()       # (M, D)
        S = self.strprs._xyz.shape[0]
        nn = self.nn_stpr_appgas.squeeze(1)                         # (M,)
        M, D = f_app.shape
        device = f_app.device
        dtype  = f_app.dtype
        out = torch.zeros(S, D, device=device, dtype=dtype)
        cnt = torch.zeros(S,    device=device, dtype=dtype)
        out.index_add_(0, nn, f_app)
        cnt.index_add_(0, nn, torch.ones(M, device=device, dtype=dtype))

        cnt_safe = cnt.clamp_min(1.0)
        f_strpr = out / cnt_safe.unsqueeze(-1)                 # (S,D)
        mask_nonempty = (cnt > 0)
        if mask_nonempty.any():
            f_strpr[mask_nonempty] = F.normalize(f_strpr[mask_nonempty], dim=-1)
        text = F.normalize(self.text_feats[:2], dim=-1)            # (2,D)
        scores = f_strpr @ text.T                              # (S,2), cos(f_strpr, t_c)
        logits = scores / tau
        pi = F.softmax(logits, dim=-1)             
        pi_leaf, pi_branch = pi[:, 0], pi[:, 1]
        ce = -( (1.0 - p_geo) * torch.log(pi_leaf) + p_geo * torch.log(pi_branch) )  # (S,)
        return ce.mean()

    def remove_low_opacity(self,threshold=0.2):
        opacity = self.get_opacity.detach().squeeze(1)
        self.tmp_radii = self.max_radii2D
        keep_mask = opacity >= threshold
        self.prune_points(~keep_mask)
        num_low_opacity = (~keep_mask).sum().item()
        print(f"Removed {num_low_opacity} Gaussians with opacity < {threshold}.")
    
    def bind_global(self):
        appgas = self.appgas._xyz.detach()
        center = self.strprs.get_xyz
        n_strpr = center.shape[0]
        rotation = build_rotation(self.strprs.get_rotation)
        u = rotation[:,:, 0]
        h = self.strprs.get_scaling[:, 0] * 1.5
        top, bottom = center + h[:, None] * u, center - h[:, None] * u
        p = torch.vstack((top, bottom)).reshape(-1,3)  # (N,2,3)
        # chamfer distance between appgas and strpr endpoints
        loss_bind_global = chamfer_distance(appgas.unsqueeze(0), p.unsqueeze(0))[0]
        return loss_bind_global.mean()

    def align_strpr_main_axis(self):
        # obtain nearest neightbor of strpr
        self.reset_neighbors()
        nn = self.knn_idx[:, 1]  # (N_strpr,)

        strpr_rot = build_rotation(self.get_rotation.detach())
        main_axis = strpr_rot[:,:,0]  # (N_strpr,3)
        main_axis_nn = strpr_rot[nn,:,0]  # (N_strpr,3)       
        cosine_sim = (main_axis * main_axis_nn).sum(dim=-1)  # (N_strpr,)
        loss_axis = (cosine_sim**2).mean()
        return loss_axis


    def compute_color_contrast_loss(self, viewpoint_camera):
        """
        Color-contrast loss that updates ONLY StrPr labels.
        L_col = E_a [ p_s * ||c_a - cbar_leaf||^2 + (1-p_s) * ||c_a - cbar_branch||^2 ],
        where s = NN(StrPr) for app a. Gradients flow only to self.strprs.label.
        Returns: loss_col, (cbar_leaf, cbar_branch)
        """
        # ---- 1) 取 AppGa 颜色并可选归一化（到 [0,1] 或均值/方差归一化） ----
        # 假设 self.appgas.get_rgb() -> (N_app,3) in [0,1]; 若不是请替换
        # color = self.get_features # (N_app, 3)
        shs_view = self.get_features.transpose(1, 2).view(-1, 3, (self.max_sh_degree+1)**2)
        dir_pp = (self.get_xyz - viewpoint_camera.camera_center.repeat(self.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        color = colors_precomp.detach()
        prob_branch = self.label_activation(self.label)  # (N_strpr,)
        prob_leaf   = 1.0 - prob_branch
        # Use FIXED reference colours (cached from the colour-calibrated init) so the loss
        # is a stable attractor toward the correct class, instead of chasing running means
        # that drift as the labels move (which collapses the classification).
        if getattr(self, '_cbar_branch', None) is None:
            branch_mask = (prob_branch > 0.5).squeeze()
            self._cbar_branch = color[branch_mask].mean(dim=0).detach()
            self._cbar_leaf = color[~branch_mask].mean(dim=0).detach()
        mean_branch_color = self._cbar_branch
        mean_leaf_color = self._cbar_leaf
        loss_col = prob_branch.unsqueeze(1) * ((color - mean_branch_color)**2).sum(dim=-1, keepdim=True) + \
                   prob_leaf.unsqueeze(1)   * ((color - mean_leaf_color)**2).sum(dim=-1, keepdim=True)
        loss_col = loss_col.mean()              
        return loss_col
    


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
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import KMeans
import faiss
from pytorch3d.transforms import quaternion_to_matrix, quaternion_invert, quaternion_apply, matrix_to_quaternion
from pytorch3d.ops import knn_points, estimate_pointcloud_normals
import networkx as nx
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement
import trimesh
from utils.gs_utils import fit_cylinder_ransac, estimate_gs_para_from_cluster, branch_to_cylinder, leaf_to_disk, stpr_to_cylinder, gs_to_disk_distance, gs_to_cylinder_distance, stpr_to_disk, build_edge, build_mst_from_endpoints,save_mst_ply
import time
from utils.loss_utils import mst_loss
from pytorch3d.loss import chamfer_distance
import open3d as o3d

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

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

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default", device=None):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._mask = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.device = device
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.knn_to_track = 4
        self.setup_functions()
        self.knn_dists = None
        self.knn_idx = None
        self.n_points = None
        self.nn_stpr_appgs = None
        self.structure_gs = None
        self.appgs = None
        self.leaf_disks = None
        self.cylinder_mesh = None
        self.label = None
        
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._mask,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.n_points,
            self.label
            # self.appgs,
            # self.structure_gs,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._mask,
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        self.n_points,
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
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
    def get_n_points(self):
        return len(self._xyz)
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    @property
    def get_mask(self):
        return self._mask

    def opacity_regularizer(self):
        return torch.mean(self.get_opacity * (1 - self.get_opacity))
    
    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1, return_full = False):
        if not return_full:
            return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
        else:
            cov = self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
            cov_full = torch.zeros((cov.shape[0], 3, 3), device=self.device)
            cov_full[:, 0, 0] = cov[:, 0]
            cov_full[:, 1, 1] = cov[:, 3]
            cov_full[:, 2, 2] = cov[:, 5]
            cov_full[:, 0, 1] = cov[:, 1]
            cov_full[:, 0, 2] = cov[:, 2]
            cov_full[:, 1, 2] = cov[:, 4]
            cov_full[:, 1, 0] = cov[:, 1]
            cov_full[:, 2, 0] = cov[:, 2]
            cov_full[:, 2, 1] = cov[:, 4]
            return cov_full

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=self.device)
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=self.device))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self._mask = torch.ones((self._xyz.shape[0],), dtype=torch.float, device=self.device)
        self.pretrained_exposures = None
        self.n_points = len(self._xyz)
        exposure = torch.eye(3, 4, device=self.device)[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))
        self.label = nn.Parameter(torch.zeros((self.get_xyz.shape[0],), dtype=torch.float, device=self.device).requires_grad_(False))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr_stpr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr_stpr, "name": "rotation"},
            {'params': [self.label], 'lr': training_args.exposure_lr_init, "name": "label"},
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

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
        return l

    def save_ply(self, path,save_only_branch=False):
        mkdir_p(os.path.dirname(path))
        if save_only_branch:
            branch_mask = torch.sigmoid(self.label) < 0.5
            xyz = self._xyz[branch_mask].detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self._features_dc[branch_mask].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest[branch_mask].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self._opacity[branch_mask].detach().cpu().numpy()
            scale = self._scaling[branch_mask].detach().cpu().numpy()
            rotation = self._rotation[branch_mask].detach().cpu().numpy()
            label = self.label[branch_mask].detach().unsqueeze(1).cpu().numpy()
        else:
            xyz = self._xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self._opacity.detach().cpu().numpy()
            scale = self._scaling.detach().cpu().numpy()
            rotation = self._rotation.detach().cpu().numpy()
            label = self.label.detach().cpu().unsqueeze(1).numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        dtype_full.append(('label', 'f4'))
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation,label), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
    
    def reset_opacity_stpr(self):
        # set opacity to 1
        opacities_new = self.inverse_opacity_activation(torch.ones_like(self.get_opacity)*0.5)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        vertex_groups = plydata['vertex'].data.dtype.names
        if "label" in vertex_groups:
            label = np.asarray(plydata.elements[0]["label"])[..., np.newaxis]
        else:
            label = np.zeros((xyz.shape[0], 1), dtype=np.float32)
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

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
        self.label = nn.Parameter(torch.tensor(label, dtype=torch.float, device=self.device).requires_grad_(True))

        # self.label = nn.Parameter(torch.zeros((self.get_xyz.shape[0],), dtype=torch.float, device=self.device).requires_grad_(False))

        self.active_sh_degree = self.max_sh_degree

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

    def prune_points(self, mask,flag=Literal['app','stpr']):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.label = optimizable_tensors["label"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

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

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_label,flag):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "label": new_label,}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.label = optimizable_tensors["label"]


        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        assert len(self.tmp_radii) == self.get_xyz.shape[0]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)
        

    def densify_and_split(self, grads, grad_threshold, scene_extent, flag=Literal['app','stpr'],N=2):
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
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)
        new_label = [lbl for lbl, m in zip(self.label, selected_pts_mask) if m]
        new_label = new_label * N
        if len(new_label)>0:
            new_label = torch.stack(new_label, dim=0)
        else:
            new_label = torch.empty(0,device=self.device)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii,new_label,flag)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=self.device, dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent,flag=Literal['app', 'stpr']):
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
        new_tmp_radii = self.tmp_radii[selected_pts_mask]
        new_label = [lbl for lbl, m in zip(self.label, selected_pts_mask) if m]
        if len(new_label)>0:
            new_label = torch.stack(new_label, dim=0)
        else:
            new_label = torch.empty(0,device=self.device)                
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii,new_label,flag)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii, flag=Literal['app','stprs'],only_prune=False,size_threshold_small=None,nn_stpr_app=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        if not only_prune :
            self.densify_and_clone(grads, max_grad, extent,flag)
            self.densify_and_split(grads, max_grad, extent,flag)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        prune_low_opacity_count = prune_mask.sum()
        prune_large_count = 0
        prune_small_count = 0

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            prune_large_count = torch.logical_or(big_points_vs, big_points_ws).sum()
        if size_threshold_small:
            small_points_ws = self.get_scaling.max(dim=1).values < 0.005 * extent
            prune_mask = torch.logical_or(prune_mask, small_points_ws)
            prune_small_count = small_points_ws.sum()
        if nn_stpr_app is not None: #TODO: prune by min bind numq
            nn = nn_stpr_app
            parent_id = nn.squeeze(1)
            p = self._xyz.shape[0]
            count = torch.bincount(parent_id, minlength=p)
            prune_mask_nn = count < 3
            prune_mask = torch.logical_or(prune_mask, prune_mask_nn)
        if flag == 'app':
            self.prune_points(prune_mask,flag=flag)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None
        print(f"Pruned {prune_low_opacity_count} points with low opacity, {prune_large_count} points with large screen size, {prune_small_count} points with small screen size.")
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def build_stprs_from_gs(self, num_clusters=100,method: Literal['kmeans', '3dgs'] = '3dgs',denoise=True):
        """
        structural primitives (StPrs) from optimized Gaussians by clustering them.
        param num_clusters: Number of clusters for grouping Gaussians into structural primitives.
        """
        print(f"Building {num_clusters} structural primitives using {method} method.")
        # Extract Gaussian properties
        xyz = self._xyz.detach().cpu().numpy()  # Gaussian positions
        scaling = self.get_scaling.detach().cpu().numpy()  # Gaussian scales
        rotation = self.get_rotation.detach().cpu().numpy()  # Gaussian rotations
        covariance = self.get_covariance().detach().cpu().numpy()  # Gaussian covariance matrices
        feature_dc = self.get_features_dc.detach().cpu().numpy()  # Gaussian features
        feature_rest = self.get_features_rest.detach().cpu().numpy()  # Gaussian features
        if denoise:
            min_pts  = 3
            radius = 0.2
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd_clean, idx_mask = pcd.remove_radius_outlier(
                        nb_points=min_pts,
                        radius=radius)
            xyz = xyz[idx_mask]
            scaling = scaling[idx_mask]
            rotation = rotation[idx_mask]
            covariance = covariance[idx_mask]
            feature_dc = feature_dc[idx_mask]
            feature_rest = feature_rest[idx_mask]
        # Initialize lists for StPr parameters
        stpr_positions = []
        stpr_scales = []
        stpr_rotations = []
        stpr_features_dc = []
        stpr_features_rest = []
        branch_positions = []
        branch_scales = []
        branch_rotations = []
        branch_points_all = []
        leaf_positions = []
        leaf_scales = []
        leaf_rotations = []
        branch_quat = []
        leaf_quat = []
        branch_feature_dc = []
        branch_feature_rest = []
        leaf_feature_dc = []
        leaf_feature_rest = []
        label_list = []
        branch_index = []
        leaf_index = []
        surf_rotations = []
        if method == 'kmeans':
            # Flatten covariance matrices for clustering
            # add a noise filter to remove noise points
            min_pts  = 6
            radius = 0.05
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd_clean, idx_mask = pcd.remove_radius_outlier(
                        nb_points=min_pts,
                        radius=radius)
            covariances_flat = covariance.reshape(covariance.shape[0], -1)
            feature_vectors = np.hstack((xyz[idx_mask], covariances_flat[idx_mask]))  # Include covariance in clustering
            # Use faiss for clustering
            kmeans = faiss.Kmeans(d=feature_vectors.shape[1], k=num_clusters, niter=20, nredo=5,gpu=True)
            kmeans.train(feature_vectors.astype(np.float32))
            cluster_labels = kmeans.index.search(feature_vectors.astype(np.float32), 1)[1].flatten()

            for cluster_id in range(num_clusters):
                cluster_points = xyz[cluster_labels == cluster_id]
                cluster_scales = scaling[cluster_labels == cluster_id]
                cluster_rotations = rotation[cluster_labels == cluster_id]

                if len(cluster_points) > 1:
                    # Compute PCA for scaling & rotation estimation
                    mean,stpr_rot, stpr_scale = estimate_gs_para_from_cluster(cluster_points)

                stpr_positions.append(mean)
                stpr_scales.append(stpr_scale)
                stpr_rotations.append(stpr_rot)
        elif method == '3dgs':
            """
            Build structural primitives (StPrs) from optimized Gaussians using 3D Gaussian clustering.
            Using segmented leaf&branch points information for initialize leaf stpr(disk) and branch stpr(cylinder).
            test for soft label 1:leaf, 0:branch
            """

            label_leaf, label_branch, labels = fit_cylinder_ransac(xyz, save_ply=True)
            index = 0
            for i,label in enumerate(np.unique(labels)):
                if label in label_leaf:
                    # build disk gs
                    leaf_points = xyz[labels == label]
                    mean,stpr_rot, stpr_scale,rot_cylinder,rot_disk = estimate_gs_para_from_cluster(leaf_points,test_flag=False)
                    # stpr_scale[2] = 1e-6
                    stpr_positions.append(mean)
                    stpr_scales.append(stpr_scale)
                    stpr_rotations.append(stpr_rot)
                    feature_dc_stpr = feature_dc[labels == label].mean(axis=0)
                    feature_rest_stpr = feature_rest[labels == label].mean(axis=0)
                    stpr_features_dc.append(feature_dc_stpr)
                    stpr_features_rest.append(feature_rest_stpr)
                    leaf_positions.append(mean)
                    leaf_scales.append(stpr_scale)
                    leaf_quat.append(stpr_rot)
                    leaf_rotations.append(rot_disk)
                    leaf_feature_dc.append(feature_dc_stpr)
                    leaf_feature_rest.append(feature_rest_stpr)
                    surf_rotations.append(rot_disk)
                    # label.append('leaf')
                    prior = torch.tensor(0.51, device=self.device)
                    label_list.append(torch.logit(prior))
                    leaf_index.append(index)
                    index += 1
                    
                elif label in label_branch:
                    # build 3dgs
                    branch_points = xyz[labels == label]
                    mean,stpr_rot, stpr_scale,rot_cylinder,rot_disk = estimate_gs_para_from_cluster(branch_points,test_flag=False)
                    stpr_scales.append(stpr_scale)
                    stpr_rotations.append(stpr_rot)
                    stpr_positions.append(mean)
                    feature_dc_stpr = feature_dc[labels == label].mean(axis=0)
                    feature_rest_stpr = feature_rest[labels == label].mean(axis=0)
                    stpr_features_dc.append(feature_dc_stpr)
                    stpr_features_rest.append(feature_rest_stpr)
                    # record branch stprs for later use
                    branch_positions.append(mean)
                    branch_scales.append(stpr_scale)
                    branch_rotations.append(rot_cylinder)
                    surf_rotations.append(rot_cylinder)
                    branch_quat.append(stpr_rot)
                    branch_points_all.append(branch_points)
                    branch_feature_dc.append(feature_dc_stpr)
                    branch_feature_rest.append(feature_rest_stpr)
                    prior = torch.tensor(0.49, device=self.device)
                    label_list.append(torch.logit(prior))
                    # label.append('branch')
                    branch_index.append(index)
                    index += 1
                    # build cylinder gs
            # mesh_cylinder = branch_to_cylinder( branch_points=np.vstack(branch_points_all),branch_positions=branch_positions,
            #                    branch_scales=branch_scales, branch_rotations=branch_rotations) # list of open3d mesh
            # leaf_disk = leaf_to_disk(leaf_positions=leaf_positions, leaf_scales=leaf_scales, leaf_rotations=leaf_rotations,save_flag=False) # list of open3d mesh
            
            # self.appgs = self.build_appgs_from_stprs(mesh_cylinder,branch_scales,branch_quat, branch_feature_dc,branch_feature_rest,
            #                                          leaf_disk,leaf_scales, leaf_quat, leaf_feature_dc, leaf_feature_rest,
            #                                          branch_label=branch_index, leaf_label=leaf_index,
            #                                          samples_per_branch=10, samples_per_leaf=10,)
                                                     
            self.branch_label = branch_index
            self.leaf_label = leaf_index
        else:
            raise ValueError("Unknown clustering method. Use 'kmeans' or '3dgs'.")

        # Convert lists to tensors
        # stpr_rotations = np.roll(np.array(stpr_rotations), 1, axis=1)
        stpr_positions = torch.tensor(np.array(stpr_positions), dtype=torch.float, device=self.device)
        stpr_scales = torch.tensor(np.array(stpr_scales), dtype=torch.float, device=self.device)
        stpr_rotations = torch.tensor(stpr_rotations, dtype=torch.float, device=self.device)
        stpr_features_dc = torch.tensor(np.array(stpr_features_dc), dtype=torch.float, device=self.device)
        stpr_features_rest = torch.tensor(np.array(stpr_features_rest), dtype=torch.float, device=self.device)
        label_tensor = torch.stack(label_list, dim=0)
        # stpr_opacities = torch.tensor(stpr_opacities, dtype=torch.float, device=self.device)
        stpr_scales[torch.isnan(stpr_scales)] = 0.1
        stpr_scales = torch.log(stpr_scales)
        stpr_sur_rots = torch.tensor(np.array(surf_rotations), dtype=torch.float, device=self.device)
        # stpr_opacities = torch.tensor(stpr_opacities, dtype=torch.float, device=self.device)
        stpr_opacities = self.inverse_opacity_activation( 0.5* torch.ones((stpr_positions.shape[0], 1),device=self.device))
        # Initialize a new GaussianModel for StPrs and return it
        self.structure_gs = GaussianModel(sh_degree=self.max_sh_degree, optimizer_type=self.optimizer_type, device=self.device)
        self.structure_gs._xyz = nn.Parameter(stpr_positions.requires_grad_(True))
        self.structure_gs._scaling = nn.Parameter(stpr_scales.requires_grad_(True))
        self.structure_gs._rotation = nn.Parameter(stpr_rotations.requires_grad_(True))
        self.structure_gs._features_dc = nn.Parameter(stpr_features_dc.requires_grad_(True))
        self.structure_gs._features_rest = nn.Parameter(stpr_features_rest.requires_grad_(True))
        self.structure_gs._opacity = nn.Parameter(stpr_opacities.requires_grad_(True))
        self.structure_gs.max_radii2D = torch.zeros((stpr_positions.shape[0]), device=self.device)
        self.structure_gs._mask = torch.ones((stpr_positions.shape[0],), dtype=torch.float, device=self.device)
        self.structure_gs.exposure_mapping = self.exposure_mapping
        self.structure_gs.pretrained_exposures = None
        self.structure_gs.surf_rotations = stpr_sur_rots
        self.structure_gs.label = nn.Parameter(label_tensor.requires_grad_(True))
        exposure = self._exposure.detach()
        self.structure_gs._exposure = nn.Parameter(exposure.requires_grad_(True))
        print(f"Initialized {len(stpr_positions)} Structural Primitives (StPrs) from Gaussian clustering.")

        self.structure_gs.scale_prior = self.structure_gs.get_scaling.max(dim=1).values
        return self.structure_gs

    
    def build_appgs_from_stprs(self,num_sample=10):
        """
        Build Appearance Gaussians (AppGs) from the structural primitives (StPrs).
        """
        self.build_surface()
        positions = []
        scales= [] 
        rotations = []
        new_features_dc = []
        new_features_rest = []
        label = []
        app_stpr_nn = []
        mesh_cylinder_list = self.cylinder_mesh_list
        leaf_disk_list = self.disk_mesh_list
        leaf_mask = torch.sigmoid(self.structure_gs.label) >= 0.5
        branch_mask = torch.sigmoid(self.structure_gs.label) < 0.5
        leaf_scales = self.structure_gs.get_scaling[leaf_mask]
        branch_scales = self.structure_gs.get_scaling[branch_mask]
        leaf_rotations = self.structure_gs.get_rotation[leaf_mask]
        branch_rotations = self.structure_gs.get_rotation[branch_mask]
        leaf_feature_dc = self.structure_gs.get_features_dc[leaf_mask]
        branch_feature_dc = self.structure_gs.get_features_dc[branch_mask]
        leaf_feature_rest = self.structure_gs.get_features_rest[leaf_mask]
        branch_feature_rest = self.structure_gs.get_features_rest[branch_mask]
        leaf_index = torch.where(leaf_mask)[0]
        branch_index = torch.where(branch_mask)[0]
        if mesh_cylinder_list is not None:
            for i,cylinder in enumerate(mesh_cylinder_list):
                branch_pcd = cylinder.sample_points_uniformly(num_sample)
                pos = torch.tensor(np.asarray(branch_pcd.points)).float()
                scale =(branch_scales[i].detach()/num_sample).unsqueeze(0).repeat(pos.shape[0], 1)
                rot = branch_rotations[i].detach().unsqueeze(0).repeat(pos.shape[0], 1)
                feature_dc = branch_feature_dc[i].detach().repeat(pos.shape[0], 1)
                feature_rest = branch_feature_rest[i].detach().repeat(pos.shape[0], 1)
                positions.append(pos)
                scales.append(scale)
                rotations.append(rot)
                new_features_dc.append(feature_dc)
                new_features_rest.append(feature_rest)
                parent_id = branch_index[i]
                logit = torch.logit(torch.tensor(0.5, device=self.device)) # 0.2
                labels = [logit] * pos.shape[0]
                label.extend(labels)
                app_index = torch.tensor(branch_index[i]).unsqueeze(0).repeat(pos.shape[0], 1)
                app_stpr_nn.extend(app_index)


        if leaf_disk_list is not None:
            for i,disk in enumerate(leaf_disk_list):
                leaf_pcd = disk.sample_points_uniformly(num_sample)
                pos = torch.tensor(np.asarray(leaf_pcd.points)).float()
                scale= (leaf_scales[i].detach()/num_sample).unsqueeze(0).repeat(pos.shape[0], 1)
                scale[:,2] = 1e-6
                # rot_quat = matrix_to_quaternion(leaf_rotations[i])
                rot = leaf_rotations[i].detach().unsqueeze(0).repeat(pos.shape[0], 1)
                feature_dc = leaf_feature_dc[i].detach().repeat(pos.shape[0], 1)
                feature_rest = leaf_feature_rest[i].detach().repeat(pos.shape[0], 1)
                positions.append(pos)
                scales.append(scale)
                rotations.append(rot)
                new_features_dc.append(feature_dc)
                new_features_rest.append(feature_rest)
                labels = [torch.logit(torch.tensor(0.5, device=self.device))] * pos.shape[0]  # 0.8
                label.extend(labels)
                app_index = torch.tensor(leaf_index[i]).unsqueeze(0).repeat(pos.shape[0], 1)
                app_stpr_nn.extend(app_index)


        num_total_samples = torch.vstack(positions).shape[0]
        new_opacities = self.inverse_opacity_activation(0.5* torch.ones((num_total_samples, 1),device=self.device))
        appgs_features_dc = torch.vstack(new_features_dc)
        # appgs_features_rest = torch.tensor(np.array(new_features_rest), dtype=torch.float, device=self.device) 
        appgs_features_rest = torch.vstack(new_features_rest) 
        appgs_features_dc = appgs_features_dc.reshape(-1, 1, 3)
        appgs_features_rest = appgs_features_rest.reshape(-1,15,3)
        label = torch.stack(label)
        self.appgs = GaussianModel(sh_degree=self.max_sh_degree, optimizer_type=self.optimizer_type, device=self.device)
        self.appgs._xyz = nn.Parameter(torch.vstack(positions).to(self.device).requires_grad_(True))
        self.appgs._scaling = nn.Parameter(torch.log(torch.vstack((scales))).float().to(self.device).requires_grad_(True))
        self.appgs._rotation = nn.Parameter(torch.vstack(rotations).float().to(self.device).requires_grad_(True))
        self.appgs._features_dc = nn.Parameter(appgs_features_dc.requires_grad_(True))
        self.appgs._features_rest = nn.Parameter(appgs_features_rest.requires_grad_(True))
        self.appgs._opacity = nn.Parameter(new_opacities.requires_grad_(True))
        self.appgs.max_radii2D = torch.zeros((num_total_samples), device=self.device)
        self.appgs._mask = torch.ones((num_total_samples), dtype=torch.float, device=self.device)
        self.appgs.exposure_mapping = self.exposure_mapping
        self.appgs.pretrained_exposures = None
        self.appgs.label = nn.Parameter(label.requires_grad_(True))
        self.nn_stpr_appgs = torch.stack(app_stpr_nn)
        exposure = self._exposure.detach()
        self.appgs._exposure = nn.Parameter(exposure.requires_grad_(True))
        print(f"Initialized {num_total_samples} Appearance Gaussians (AppGs) from StPrs.")
        return self.appgs

    def build_appgs_from_cylinder(self, cylinder_params, num_samples=50):
        """
        Build Appearance Gaussians (AppGs) by uniformly sampling points on the surface of the cylinder.
        
        Args:
            cylinder_params (dict): Dictionary containing cylinder parameters (center, axis, radius, height).
            num_samples (int): Number of samples to generate.
        
        Returns:
            GaussianModel: Initialized AppGs.
        """
        # Extract cylinder parameters
        C = cylinder_params["center"]  # (N, 3)
        A = cylinder_params["axis"]    # (N, 3)
        R = cylinder_params["radius"]  # (N,)
        H = cylinder_params["height"]  # (N,)

        # Normalize the axis vector
        A = A / torch.norm(A, dim=-1, keepdim=True)

        # Sample points on the top and bottom circles
        num_circle_samples = num_samples // 3
        theta = torch.linspace(0, 2 * torch.pi, num_circle_samples, device=self.device)  # (num_circle_samples,)
        circle_x = torch.cos(theta) * R[:, None]  # (N, num_circle_samples)
        circle_y = torch.sin(theta) * R[:, None]  # (N, num_circle_samples)

        # Compute two orthogonal vectors to the axis
        v1 = torch.tensor([1, 0, 0], device=self.device).repeat(A.shape[0], 1)
        v1 = v1 - (torch.sum(v1 * A, dim=-1, keepdim=True) * A)
        v1 = v1 / torch.norm(v1, dim=-1, keepdim=True)

        v2 = torch.cross(A, v1)  # Get the second orthogonal vector

        # Compute the points on the circles
        top_center = C + (H[:, None] / 2) * A
        bottom_center = C - (H[:, None] / 2) * A

        circle_points_top = top_center[:, None, :] + circle_x[:, :, None] * v1[:, None, :] + circle_y[:, :, None] * v2[:, None, :]
        circle_points_bottom = bottom_center[:, None, :] + circle_x[:, :, None] * v1[:, None, :] + circle_y[:, :, None] * v2[:, None, :]

        # Sample points on the side surface
        num_side_samples = num_samples - 2 * num_circle_samples
        side_theta = torch.linspace(0, 2 * torch.pi, num_side_samples, device=self.device)
        z_offsets = torch.linspace(-0.5, 0.5, num_side_samples, device=self.device) * H[:, None]

        side_x = torch.cos(side_theta) * R[:, None]
        side_y = torch.sin(side_theta) * R[:, None]
        side_z = z_offsets

        side_points = C[:, None, :] + side_x[:, :, None] * v1[:, None, :] + side_y[:, :, None] * v2[:, None, :] + side_z[:, :, None] * A[:, None, :]

        # Combine all sampled points
        sampled_points = torch.cat([circle_points_top, circle_points_bottom, side_points], dim=1).reshape(-1, 3)

        # Initialize Appearance Gaussians (AppGs)
        num_total_samples = sampled_points.shape[0]
        scaling_st = self.structure_gs._scaling
        scales = scaling_st.unsqueeze(1).repeat(1, num_samples, 1) / num_samples

        scales= scales.reshape(-1,3)
        # set all scale to -1 
        scales = torch.log(scales)
        scales[torch.isnan(scales)] = -3
        quaternion_st = self.structure_gs._rotation
        rots = quaternion_st.unsqueeze(1).repeat(1, num_samples, 1)
        rots = rots.reshape(-1,4)
        # init features
        features_dc = self.structure_gs._features_dc.unsqueeze(1).repeat(1, num_samples, 1, 1).reshape(-1,1,3)
        # assign features to each gaussian from the same structure
        features_rest = self.structure_gs._features_rest.unsqueeze(1).repeat(1, num_samples, 1, 1).reshape(-1,15,3)
        # init opacity
        # new_opacities = self.structure_gs._opacity.unsqueeze(1).repeat(1, samples_per_stgs, 1).reshape(-1,1)    
        new_opacities = self.inverse_opacity_activation(0.1 * torch.ones((sampled_points.shape[0], 1), dtype=torch.float, device=self.device))
        # new_opacities = torch.ones((sampled_points.shape[0], 1), dtype=torch.float, device=self.device)
        self.appgs = GaussianModel(sh_degree=self.max_sh_degree, optimizer_type=self.optimizer_type, device=self.device)
        self.appgs._xyz = nn.Parameter(sampled_points.requires_grad_(True))
        self.appgs._scaling = nn.Parameter(scales.requires_grad_(True))
        self.appgs._rotation = nn.Parameter(rots.requires_grad_(True))
        self.appgs._features_dc = nn.Parameter(features_dc.requires_grad_(True))
        self.appgs._features_rest = nn.Parameter(features_rest.requires_grad_(True))
        self.appgs._opacity = nn.Parameter(new_opacities.requires_grad_(True))
        self.appgs.max_radii2D = torch.zeros((sampled_points.shape[0]), device=self.device)
        self.appgs._mask = torch.ones((num_total_samples,), dtype=torch.float, device=self.device)
        self.appgs.exposure_mapping = self.exposure_mapping
        self.appgs.pretrained_exposures = None
        exposure = self._exposure.detach()
        self.appgs._exposure = nn.Parameter(exposure.requires_grad_(True))

        # print(f"Initialized {num_total_samples} Appearance Gaussians (AppGs) from Cylinder Surface.")
        return self.appgs

    def compute_gaussian_overlap_with_neighbors(
        self, 
        neighbor_idx,
        use_gaussian_center_only=True,
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
    
    def compute_gaussian_separation_with_neighbors(self, neighbor_idx):
        nn_idx = neighbor_idx[:, 1]
        points = self._xyz
        points_nn = points[nn_idx]
        radius  = 3*self.get_scaling[:, 0]             # (P,)

        distances = (points- points_nn).norm(dim=-1).clamp(min=1e-8)
        thr = 0.8 * torch.maximum(radius,radius[nn_idx])
        sep = torch.nn.functional.relu(thr-distances)
        loss_sep =sep.mean()
        return loss_sep
    
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

    def get_normals(self, estimate_from_points=False, neighborhood_size:int=32):
        """Returns the normals of the Gaussians.

        Args:
            estimate_from_points (bool, optional): _description_. Defaults to False.
            neighborhood_size (int, optional): _description_. Defaults to 32.

        Returns:
            _type_: _description_
        """
        if estimate_from_points:
            normals = estimate_pointcloud_normals(
                self.points[None], #.detach(), 
                neighborhood_size=neighborhood_size,
                disambiguate_directions=True
                )[0]
        else:
            if self.binded_to_surface_mesh:
                normals = torch.nn.functional.normalize(self.surface_mesh.faces_normals_list()[0], dim=-1).view(-1, 1, 3)
                normals = normals.expand(-1, self.n_gaussians_per_surface_triangle, -1).reshape(-1, 3)
            else:
                normals = self.get_smallest_axis()
        return normals
    
    def get_neighbors_of_random_points(self, num_samples):
        if num_samples >= 0:
            sampleidx = torch.randperm(len(self._xyz), device=self.device)[:num_samples]        
            return self.knn_idx[sampleidx]
        else:
            return self.knn_idx
    
    def get_local_variance(self, values:torch.Tensor, neighbor_idx:torch.Tensor):
        """_summary_

        Args:
            values (_type_): Shape is (n_points, n_values)
            neighbor_idx (_type_): Shape is (n_points, n_neighbors)
        """
        neighbor_values = values[neighbor_idx]  # Shape is (n_points, n_neighbors, n_values)
        return (neighbor_values - neighbor_values.mean(dim=1, keepdim=True)).pow(2).sum(dim=-1).mean(dim=1)
    
    def get_local_distance2(
        self, 
        values:torch.Tensor, 
        neighbor_idx:torch.Tensor, 
        weights:torch.Tensor=None,
        ):
        """_summary_

        Args:
            values (torch.Tensor): Shape is (n_points, n_values)
            neighbor_idx (torch.Tensor): Shape is (n_points, n_neighbors)
            weights (torch.Tensor, optional): Shape is (n_points, n_neighbors). Defaults to None.

        Returns:
            _type_: _description_
        """
        
        neighbor_values = values[neighbor_idx]  # Shape is (n_points, n_neighbors, n_values)
        distance2 = neighbor_values[:, 1:] - neighbor_values[:, :1]  # Shape is (n_points, n_neighbors-1, n_values)
        distance2 = distance2.pow(2).sum(dim=-1)  # Shape is (n_points, n_neighbors-1)
        
        if weights is not None:
            distance2 = distance2 * weights

        return distance2.mean(dim=1)  # Shape is (n_points,)
    
    def reset_neighbors(self):
        # Compute KNN               
        with torch.no_grad():
            knns = knn_points(self._xyz[None], self._xyz[None], K=self.knn_to_track)
            self.knn_dists = knns.dists[0]
            self.knn_idx = knns.idx[0]

    def update_nn_between_appgs_and_stprs(self):
        # # compute mahalanobis distance between appgs and stprs
        # appgs_xyz = self.appgs._xyz.detach()
        # stprs_xyz = self.structure_gs._xyz.detach()
        # stprs_cov = self.structure_gs.get_covariance(return_full=True)
        # # 2) regularize & invert covariances for numeric stability
        # D = stprs_cov.shape[-1]
        # eps = 1e-6
        # eye = torch.eye(D, device=stprs_cov.device).unsqueeze(0)  # (1, D, D)
        # stprs_cov = stprs_cov + eps * eye                        # (N_str, D, D)
        # stprs_cov_inv = torch.linalg.inv(stprs_cov)              # (N_str, D, D)

        # # 3) build difference tensor: (N_app, N_str, 3)
        # diffs = appgs_xyz.unsqueeze(1) - stprs_xyz.unsqueeze(0)

        # # 4) compute squared Mahalanobis distances:
        # #    md2[a, i] = diffs[a,i,:] @ stprs_cov_inv[i] @ diffs[a,i,:]^T
        # md2 = torch.einsum("aij,ijk,aik->ai", diffs, stprs_cov_inv, diffs)

        # # 5) pick the nearest StrPr for each AppG
        # nn_index = md2.argmin(dim=1)    # (N_app,)
        # self.nn_stpr_appgs = nn_index
        app_xyz = self.appgs._xyz.detach()                     # (N_app, 3)
        stpr_xyz = self.structure_gs._xyz.detach()             # (N_str, 3)
        scaling = self.structure_gs.get_scaling.detach()  # (N_str, 3)
        quats   = self.structure_gs.get_rotation.detach() # (N_str, 4)

        # 2. 阈值截断，确保用来除法的标准差不为 0
        eps = 1e-4
        # 假设 scaling 本意是高斯在每个轴方向上的 σ²，先取开根号得到 σ
        sigma = torch.sqrt(torch.clamp(scaling, min=eps))      # (N_str, 3)

        # 3. 广播到 (N_app, N_str, ...)
        N_app, N_str = app_xyz.shape[0], stpr_xyz.shape[0]
        app_expand  = app_xyz.unsqueeze(1).expand(-1, N_str, -1)   # (N_app, N_str, 3)
        stpr_expand = stpr_xyz.unsqueeze(0).expand(N_app, -1, -1)  # (N_app, N_str, 3)
        shift       = app_expand - stpr_expand                     # (N_app, N_str, 3)

        # 4. 旋转到高斯的局部坐标系
        quats_expand = quats.unsqueeze(0).expand(N_app, -1, -1)    # (N_app, N_str, 4)
        inv_quats    = quaternion_invert(quats_expand)            # (N_app, N_str, 4)
        local_shift  = quaternion_apply(inv_quats, shift)         # (N_app, N_str, 3)

        # 5. 按轴除以 σ，得到标准化后的偏移
        inv_sigma = 1.0 / sigma.unsqueeze(0)                      # (1, N_str, 3)
        normed    = local_shift * inv_sigma                       # (N_app, N_str, 3)

        # 6. 最终取 L2 norm，得到 Mahalanobis 距离
        maha2 = normed.pow(2).sum(dim=-1)                         # (N_app, N_str)
        maha  = torch.sqrt(maha2 + 1e-8)                          # (N_app, N_str)

        # 7. 为每个 AppG 选最近的 StrPr
        nn_index = maha.argmin(dim=1)   
        self.nn_stpr_appgs = nn_index.unsqueeze(1)  # (N_app, 1)

    def get_edge_neighbors(self, k_neighbors, 
                           edges=None, triangle_vertices=None,):
        if edges is None:
            edges = self.triangle_border_edges
        if triangle_vertices is None:
            triangle_vertices = self.triangle_vertices
        
        # We select the closest edges based on the position of the edge center
        edge_centers = triangle_vertices[edges].mean(dim=-2)
        
        # TODO: Compute only for vertices with high opacity? Remove points with low opacity?
        edge_knn = knn_points(edge_centers[None], edge_centers[None], K=8)
        edge_knn_idx = edge_knn.idx[0]
        
        return edge_knn_idx

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
    
    def sample_points_in_gaussians(self, num_samples, sampling_scale_factor=1., mask=None,
                                   probabilities_proportional_to_opacity=False,
                                   probabilities_proportional_to_volume=True,):
        """Sample points in the Gaussians.

        Args:
            num_samples (_type_): _description_
            sampling_scale_factor (_type_, optional): _description_. Defaults to 1..
            mask (_type_, optional): _description_. Defaults to None.
            probabilities_proportional_to_opacity (bool, optional): _description_. Defaults to False.
            probabilities_proportional_to_volume (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
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
    
    def drop_low_opacity_points(self, opacity_threshold=0.5):
        mask = self.get_opacity[...,0] < opacity_threshold  # 1e-3, 0.5
        self.prune_points(mask)
        print(f"Dropped {mask.sum()} points with opacity below {opacity_threshold}.")
        print(f"""Remaining points: {len(self._xyz)}""")  
    
    def convert_gs_to_cylinders(self, sigma=3.0):
        cov = self.get_covariance(return_full=True)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        lambda1, lambda2, lambda3 = eigenvalues[..., 0], eigenvalues[..., 1], eigenvalues[..., 2]
        radius = (sigma * torch.sqrt(torch.maximum(lambda1, lambda2)))
        axis = eigenvectors[:, :, 2]  # (N, 3)
        height = (sigma * torch.sqrt(lambda3))
        center = self._xyz
        cylinder_params = {
            "center": center,
            "axis": axis,
            "radius": radius,
            "height": height,
        }
        return cylinder_params

    def gs_to_graph(self, k=3, filename="gaussian_graph.ply"):
        """
        Save Structural Gaussians (StPrs) as a PLY file with edges based on nearest neighbors.

        Parameters:
        - stprs_xyz (torch.Tensor): Tensor of shape (N, 3), Gaussian center positions.
        - k (int): Number of nearest neighbors to connect.
        - filename (str): Output file name for the PLY file.
        """
        # Ensure tensor is detached and converted to numpy
        xyz = self._xyz.detach().cpu().numpy()
        num_points = xyz.shape[0]

        # Compute k-nearest neighbors
        tree = cKDTree(xyz)
        distances, indices = tree.query(xyz, k=k+1)  # k+1 to include self, will remove later

        # Prepare vertex data
        vertices = [(xyz[i][0], xyz[i][1], xyz[i][2]) for i in range(num_points)]

        # Prepare edge data (store as list of tuples)
        edges = []
        for i in range(num_points):
            for j in indices[i][1:]:  # Skip self-connection (first index)
                edges.append((i, j))

        # Convert to NumPy structured arrays
        vertex_dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4")])
        edge_dtype = np.dtype([("vertex1", "i4"), ("vertex2", "i4")])

        vertex_array = np.array(vertices, dtype=vertex_dtype)
        edge_array = np.array(edges, dtype=edge_dtype)

        # Create PLY elements
        vertex_element = PlyElement.describe(vertex_array, "vertex")
        edge_element = PlyElement.describe(edge_array, "edge")

        # Save to PLY file
        PlyData([vertex_element, edge_element], text=True).write(filename)
        print(f"PLY file saved as {filename}")
        
    def gs_cylinder_distance(self, nn_index, cylinder_params):
        # Extract Gaussian parameters
        C = cylinder_params["center"][nn_index].squeeze(1)  # (M, 3) Nearest cylinder centers
        A = cylinder_params["axis"][nn_index].squeeze(1)    # (M, 3) Nearest cylinder axes
        R = cylinder_params["radius"][nn_index].squeeze(1)  # (M,) Nearest cylinder radii
        H = cylinder_params["height"][nn_index].squeeze(1)  # (M,) Nearest cylinder heights

        # Extract AppG positions
        P = self._xyz  # (M, 3) AppG center positions

        # Step 1: Project AppG centers onto the cylinder axis
        AP = P - C  # (M, 3) Vector from cylinder center to AppG
        proj_scalar = torch.sum(AP * A, dim=-1)  # (M,) Projection length along axis
        proj_point = C + proj_scalar.unsqueeze(-1) * A  # (M, 3) Projected points on the cylinder axis

        # Step 2: Compute radial distance from projected point to AppG
        radial_vector = P - proj_point  # (M, 3)
        radial_distance = torch.norm(radial_vector, dim=-1)  # (M,)
        distance_to_surface = radial_distance - R  # (M,)
        return distance_to_surface.mean()
    
    def low_freq_loss(self):
        cov = self.get_covariance(return_full=True)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        freq = 1.0 / torch.sqrt(eigenvalues+ 1e-6)
        return freq.mean() 

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

    def compute_binding_loss(self,compute_only_stpr=False):
        """ Rebuild cylinder and disk meshes from the stprs """
        if isinstance(self.nn_stpr_appgs, list):        
            nn_stpr_appgs = torch.stack(self.nn_stpr_appgs, dim=0).to(self.device)
        else:                                           
            nn_stpr_appgs = self.nn_stpr_appgs.to(self.device)        
        p_leaf = torch.sigmoid(self.structure_gs.label)
        p_leaf_app = p_leaf[nn_stpr_appgs].squeeze(1)
        p_branch = 1 - p_leaf
        p_branch_app = 1 - p_leaf_app
        if compute_only_stpr:
            app_xyz = self.appgs._xyz.detach()
        else:
            app_xyz = self.appgs._xyz
        disk_param_all,_,_ = stpr_to_disk(self.structure_gs._xyz, self.structure_gs.get_scaling, self.structure_gs.get_rotation, save_flag=False)
        dist_leaf_all   = gs_to_disk_distance(app_xyz,   nn_stpr_appgs,   disk_param_all)
        weighted_dist_leaf   = dist_leaf_all* p_leaf_app
        loss_disk= (weighted_dist_leaf**2).mean()
        cyl_param_all,_,_ = stpr_to_cylinder(self.structure_gs._xyz, self.structure_gs.get_scaling, self.structure_gs.get_rotation, save_flag=False)
        dist_branch_all = gs_to_cylinder_distance(app_xyz, nn_stpr_appgs, cyl_param_all)
        weighted_dist_branch = dist_branch_all * p_branch_app
        loss_cylinder  = (weighted_dist_branch**2).mean() 
        loss_bind = loss_disk + loss_cylinder
        dist_diff = torch.abs(dist_leaf_all - dist_branch_all)
        loss_shape = p_leaf_app*dist_diff + p_branch_app*dist_diff
        loss_bind = loss_bind + loss_shape.mean()
        return loss_bind
    
    def color_loss(self):
        p = torch.sigmoid(self.label)
        p_branch = torch.sigmoid(self.label) < 0.5
        p_leaf = torch.sigmoid(self.label) >= 0.5
        color = self.get_features.squeeze(1).detach()
        c_b = color[p_branch].mean(dim=0, keepdim=True).detach()   # (1,C)
        c_l = color[p_leaf]  .mean(dim=0, keepdim=True).detach()   # (1,C)

        # ---- 2) 对所有 StPr 计算颜色差 ----
        diff_leaf   = torch.norm(color - c_l, dim=1, keepdim=True)    # (P,1)
        diff_branch = torch.norm(color - c_b, dim=1, keepdim=True) 

        loss_color = (p.view(-1,1) * diff_leaf + (1 - p.view(-1,1)) * diff_branch).mean()
        return loss_color

    def build_surface(self):
        branch_mask = torch.sigmoid(self.structure_gs.label) < 0.5
        branch_pos   = self.structure_gs.get_xyz[branch_mask]
        branch_scale = self.structure_gs.get_scaling[branch_mask]
        branch_rot   = self.structure_gs.get_rotation[branch_mask]
        cyl_param , cylinder_mesh, cylinder_mesh_list= stpr_to_cylinder(branch_pos, branch_scale, branch_rot, save_flag=True)
        leaf_mask = torch.sigmoid(self.structure_gs.label) >= 0.5
        leaf_pos = self.structure_gs.get_xyz[leaf_mask]
        leaf_scale = self.structure_gs.get_scaling[leaf_mask]
        leaf_rot = self.structure_gs.get_rotation[leaf_mask]
        disk_param, disk_mesh,disk_list = stpr_to_disk(leaf_pos, leaf_scale, leaf_rot, save_flag=True)
        self.cylinder_mesh = cylinder_mesh
        self.disk_mesh = disk_mesh      
        self.cylinder_mesh_list = cylinder_mesh_list
        self.disk_mesh_list = disk_list  
        
    def compute_gaussian_binding_loss(self, n_samples=1, reduction=Literal['sum','mean'],method=Literal['surface', 'mahalanobis']):
        """
        Compute binding loss from appgs to structure_gs.
        This reflects how well appgs are spatially explained by stprs.

        Args:
            nn_idx: LongTensor (N, K) - stpr neighbors for each appg
            n_samples: int - how many samples to simulate per gaussian
            reduction: 'mean' | 'sum' | 'none'

        Returns:
            loss: scalar or (N,) if reduction='none'
        """
        nn_idx = self.nn_stpr_appgs
        N = len(nn_idx)

        if method == "mahalanobis":
            # 1. Sample appg points
            if n_samples == 1:
                samples = self.appgs._xyz.unsqueeze(1)  # (N, 1, 3)
            else:
                samples = self.appgs._xyz.unsqueeze(1) + quaternion_apply(
                    self.appgs._rotation.unsqueeze(1),
                    self.appgs._scaling.unsqueeze(1) * torch.randn(N, n_samples, 3, device=self.device)
                )  # (N, n_samples, 3)

            # 2. Neighbor gaussians
            nbr_xyz   = self.structure_gs._xyz[nn_idx]         # (N, K, 3)
            nbr_rot   = self.structure_gs._rotation[nn_idx]    # (N, K, 4)
            nbr_scale = self.structure_gs._scaling[nn_idx]     # (N, K, 3)

            # 3. Compute offset from appgs to neighbor centers
            offset = samples.unsqueeze(1) - nbr_xyz.unsqueeze(2)  # (N, K, n_samples, 3)

            # 4. Transform offset to local stpr space
            local_offset = quaternion_apply(
                quaternion_invert(nbr_rot).unsqueeze(2),  # (N, K, 1, 4)
                offset  # (N, K, n_samples, 3)
            )

            # 5. Normalize by inverse scale
            normed = local_offset / nbr_scale.unsqueeze(2)  # (N, K, n_samples, 3)

            # 6. Distance as binding loss
            dist = torch.norm(normed, dim=-1)  # (N, K, n_samples)

            dist = dist.mean(dim=-1)  # (N, K) → mean over samples

            # 7. Aggregate per-appg
            loss_per_appg = dist.mean(dim=-1)  # (N,)

            if reduction == 'mean':
                return loss_per_appg.mean()
            elif reduction == 'sum':
                return loss_per_appg.sum()
            else:
                return loss_per_appg  # (N,)
        elif method == "surface":
            """ approximate the distance between appg and stpr by the distance between appg and the surface of stpr """
            loss_bind = self.compute_binding_loss()
            return loss_bind

    def stpr_to_graph(self,opacity_threshold=0, anisotrpopy_threshold=5,save_mst=False):  # 0.2 ,30
        # MST for grpah extraction
        # step1: Noise filtering
        branch_mask = torch.sigmoid(self.structure_gs.label) < 0.5
        # keep = stpr_opcaity > opacity_threshold # low opacity filter
        # keep = keep.flatten()
        # leaf like filter
        anisotropy = self.structure_gs.get_scaling[:,0] / self.structure_gs.get_scaling[:,1]
        anisotropy = torch.max(anisotropy, 1.0/anisotropy)
        keep = (anisotropy > anisotrpopy_threshold).flatten()
        
        # step2: stpr to edge
        keep = keep & branch_mask
        center = self.structure_gs.get_xyz[keep] # (N,3)
        scales = self.structure_gs.get_scaling[keep]
        rot = self.structure_gs.get_rotation[keep]
        rot_matrix = quaternion_to_matrix(rot) #
        u = rot_matrix[:,:,0]
        h = self.structure_gs.get_scaling[keep][:,0] * 1.5  
        top = center + h.unsqueeze(1) * u
        bottom = center - h.unsqueeze(1) * u
        mst_edges, points = build_mst_from_endpoints(top,bottom)
        if save_mst:
            save_mst_ply(points, mst_edges)
        # graph loss 
        loss_graph = mst_loss(top,bottom,rot_matrix,mst_edges) 
        return mst_edges, points,loss_graph


    def geometric_loss(self):
        source = self.appgs._xyz.detach()
        target = self.structure_gs.sample_points_in_gaussians(num_samples=10000)
        # Compute the distance between the source and target points
        loss_geo = chamfer_distance(source.unsqueeze(0), target[0].unsqueeze(0))[0]
        return loss_geo.mean()
    
    def compute_gaussian_anisotropy(self,tau=3):
        p_leaf = torch.sigmoid(self.label)
        p_branch = 1 - p_leaf
        scales = self.get_scaling
        sx, sy = scales[:,0], scales[:,1]
        ani = torch.maximum(sx / sy, sy / sx)         # (P,)
        leaf_term   = (ani - 1.0)**2
        branch_term = torch.nn.functional.relu(tau - ani)**2
        loss_ani = p_leaf * leaf_term + p_branch * branch_term
        return loss_ani.mean()
    
    def save_stpr_app_correspondence(self, out_name, sphere_rad=0.005):
        xyz_stpr  = self.structure_gs.get_xyz      # (P,3)
        xyz_app   = self.appgs._xyz                 # (M,3)
        parent_id = self.nn_stpr_appgs.squeeze(1).cpu()             # (M,)

        P = xyz_stpr.shape[0]

        rng = torch.Generator().manual_seed(0)      
        colors_stpr = torch.rand((P, 3), generator=rng)             # (P,3)
        colors_app  = colors_stpr[parent_id]                        # (M,3)

        # build cylinder and disk for stpr
        leaf_mask = torch.sigmoid(self.structure_gs.label) >= 0.5
        branch_mask = ~leaf_mask
        cyl_param,_,cyl_mesh_list = stpr_to_cylinder(
            xyz_stpr[branch_mask], 
            self.structure_gs.get_scaling[branch_mask], 
            self.structure_gs.get_rotation[branch_mask], 
            save_flag=True)
        
        disk_param, _ ,disk_mesh_list = stpr_to_disk(
            xyz_stpr[leaf_mask], 
            self.structure_gs.get_scaling[leaf_mask], 
            self.structure_gs.get_rotation[leaf_mask], 
            save_flag=True)
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

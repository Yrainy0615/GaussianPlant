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
from pytorch3d.transforms import quaternion_to_matrix, quaternion_invert, quaternion_apply
from pytorch3d.ops import knn_points, estimate_pointcloud_normals
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


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
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
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.knn_to_track = 16
        self.setup_functions()
        self.knn_dists = None
        self.knn_idx = None
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

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

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
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
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

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
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

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

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

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

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

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
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

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def build_stprs_from_gs(self, num_clusters):
        """
        Build structural primitives (StPrs) from optimized Gaussians by clustering them.
        :param num_clusters: Number of clusters for grouping Gaussians into structural primitives.
        """
        # Extract Gaussian properties
        xyz = self._xyz.detach().cpu().numpy()  # Gaussian positions
        scaling = self.get_scaling.detach().cpu().numpy()  # Gaussian scales
        rotation = self.get_rotation.detach().cpu().numpy()  # Gaussian rotations
        covariance = self.get_covariance().detach().cpu().numpy()  # Gaussian covariance matrices

        # Flatten covariance matrices for clustering
        covariances_flat = covariance.reshape(covariance.shape[0], -1)
        feature_vectors = np.hstack((xyz, covariances_flat))  # Include covariance in clustering

        # Apply clustering to Gaussian positions and covariance
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(feature_vectors)
        cluster_labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_[:, :3]  # Extract only position components

        # Initialize lists for StPr parameters
        stpr_positions = []
        stpr_scales = []
        stpr_rotations = []

        for cluster_id in range(num_clusters):
            cluster_points = xyz[cluster_labels == cluster_id]
            cluster_scales = scaling[cluster_labels == cluster_id]
            cluster_rotations = rotation[cluster_labels == cluster_id]

            if len(cluster_points) > 1:
                # Compute PCA for scaling & rotation estimation
                mean = np.mean(cluster_points, axis=0)
                centered = cluster_points - mean
                cov = np.dot(centered.T, centered) / len(cluster_points)
                eigvals, eigvecs = np.linalg.eigh(cov)

                # Use PCA eigenvectors as orientation
                stpr_rot = R.from_matrix(eigvecs).as_quat()
                stpr_scale = np.sqrt(eigvals).clip(min=0.01)  # Ensure valid scale values
            else:
                # If only one point in cluster, use its existing rotation and scale
                mean = cluster_points[0]
                stpr_rot = cluster_rotations[0]
                stpr_scale = cluster_scales[0]

            stpr_positions.append(mean)
            stpr_scales.append(stpr_scale)
            stpr_rotations.append(stpr_rot)

        # Convert lists to tensors
        stpr_positions = torch.tensor(np.array(stpr_positions), dtype=torch.float, device="cuda")
        stpr_scales = torch.tensor(np.array(stpr_scales), dtype=torch.float, device="cuda")
        stpr_rotations = torch.tensor(np.array(stpr_rotations), dtype=torch.float, device="cuda")

        # Initialize a new GaussianModel for StPrs and return it
        self.structure_gs = GaussianModel(sh_degree=self.max_sh_degree, divide_ratio=self.divide_ratio)
        self.structure_gs._xyz = nn.Parameter(stpr_positions.requires_grad_(True))
        self.structure_gs._scaling = nn.Parameter(stpr_scales.requires_grad_(True))
        self.structure_gs._rotation = nn.Parameter(stpr_rotations.requires_grad_(True))
        self.structure_gs._features_dc = nn.Parameter(torch.zeros((stpr_positions.shape[0], 1, 3), device="cuda").requires_grad_(True))
        self.structure_gs._features_rest = nn.Parameter(torch.zeros((stpr_positions.shape[0], 15, 3), device="cuda").requires_grad_(True))
        self.structure_gs._opacity = nn.Parameter(torch.ones((stpr_positions.shape[0], 1), device="cuda").requires_grad_(True))
        self.structure_gs.max_radii2D = torch.zeros((stpr_positions.shape[0]), device="cuda")
        

        print(f"Initialized {num_clusters} Structural Primitives (StPrs) from Gaussian clustering.")
        return self.structure_gs

    
    def build_appgs_from_stprs(self, samples_per_stgs=10):
        if self.structure_gs is not None:
            points_st = self.structure_gs._xyz.detach()
            quaternions_st = self.structure_gs._rotation.detach()
            scaling_st = self.structure_gs._scaling.detach()
            gaussian_samples = points_st.unsqueeze(1) + quaternion_apply(
                quaternions_st.unsqueeze(1), 
                scaling_st.unsqueeze(1) * torch.randn(
                    points_st.shape[0], samples_per_stgs, 3, device="cuda"))
        
            # init gaussian parameters 
            new_means = gaussian_samples.reshape(-1,3)
            scales = scaling_st.unsqueeze(1).repeat(1, samples_per_stgs, 1) / samples_per_stgs
            scales[:, :,-1] = 1e-6
            scales= scales.reshape(-1,3)
            rots = quaternions_st.unsqueeze(1).repeat(1, samples_per_stgs, 1)
            rots = rots.reshape(-1,4)
            # init features
            features_dc = self.structure_gs._features_dc.detach().unsqueeze(1).repeat(1, samples_per_stgs, 1, 1).reshape(-1,1,3)
            # assign features to each gaussian from the same structure
            features_rest = self.structure_gs._features_rest.detach().unsqueeze(1).repeat(1, samples_per_stgs, 1, 1).reshape(-1,15,3)
            # init opacity
            new_opacities = self.structure_gs._opacity.unsqueeze(1).repeat(1, samples_per_stgs, 1).reshape(-1,1)    
            # Set Gaussian parameters
            self.appgs = GaussianModel(sh_degree=self.max_sh_degree, divide_ratio=self.divide_ratio)
            self.appgs._xyz = nn.Parameter(new_means.requires_grad_(True))
            self.appgs._scaling = nn.Parameter(scales.requires_grad_(True))
            self.appgs._rotation = nn.Parameter(rots.requires_grad_(True))
            self.appgs._features_dc = nn.Parameter(features_dc.requires_grad_(True))
            self.appgs._features_rest = nn.Parameter(features_rest.requires_grad_(True))
            self.appgs._opacity = nn.Parameter(new_opacities.requires_grad_(True))
            self.appgs.max_radii2D = torch.zeros((new_means.shape[0]), device="cuda")
            return self.appgs



        else: 
                print("No structure to bind to")

    def compute_gaussian_overlap_with_neighbors(
        self, 
        neighbor_idx,
        use_gaussian_center_only=True,
        n_samples_to_compute_overlap=32,
        weight_by_normal_angle=False,
        propagate_gradient_to_points_only=False,
        ):
        
        # This is used to skip the first neighbor, which is the point itself
        neighbor_start_idx = 1
        
        # Get sampled points
        point_idx = neighbor_idx[:, 0]  # (n_points, )
        n_points = len(point_idx)
        
        # Decide whether we want to propagate the gradient to the points only, or to the points and the covariance parameters
        if propagate_gradient_to_points_only:
            scaling = self.scaling.detach()
            quaternions = self.quaternions.detach()
        else:
            scaling = self.scaling
            quaternions = self.quaternions
        
        # Samples points in the corresponding gaussians
        if use_gaussian_center_only:
            n_samples_to_compute_overlap = 1
            gaussian_samples = self.points[point_idx].unsqueeze(1) + 0.  # (n_points, n_samples_to_compute_overlap, 3)
        else:
            gaussian_samples = self.points[point_idx].unsqueeze(1) + quaternion_apply(
                quaternions[point_idx].unsqueeze(1), 
                scaling[point_idx].unsqueeze(1) * torch.randn(
                    n_points, n_samples_to_compute_overlap, 3, 
                    device=self.device)
                )  # (n_points, n_samples_to_compute_overlap, 3)
        
        # >>> We will now compute the gaussian weight of all samples, for each neighbor gaussian.
        # We start by computing the shift between the samples and the neighbor gaussian centers.
        neighbor_center_to_samples = gaussian_samples.unsqueeze(1) - self.points[neighbor_idx[:, neighbor_start_idx:]].unsqueeze(2)  # (n_points, n_neighbors-1, n_samples_to_compute_overlap, 3)
        
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
        point_idx = neighbor_idx[:, 0]  # (n_points, )
        n_points = len(point_idx)
        
        # Decide whether we want to propagate the gradient to the points only, or to the points and the covariance parameters
        if propagate_gradient_to_points_only:
            scaling = self.scaling.detach()
            quaternions = self.quaternions.detach()
        else:
            scaling = self.scaling
            quaternions = self.quaternions
        
        # We compute scaling, inverse quaternions and centers for all gaussians and their neighbors
        all_scaling = scaling[neighbor_idx]
        all_invert_quaternions = quaternion_invert(quaternions)[neighbor_idx]
        all_centers = self.points[neighbor_idx]
        
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
            sampleidx = torch.randperm(len(self.points), device=self.device)[:num_samples]        
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
    
    def reset_neighbors(self, knn_to_track:int=None):
        # Compute KNN               
        with torch.no_grad():
            self.knn_to_track = knn_to_track
            knns = knn_points(self._xyz[None], self._xyz[None], K=knn_to_track)
            self.knn_dists = knns.dists[0]
            self.knn_idx = knns.idx[0]
            
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
        rotation_matrices = quaternion_to_matrix(self.quaternions)
        smallest_axis_idx = self.scaling.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
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
            scaling = self.scaling
        else:
            scaling = self.scaling[mask]
        
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
        
        random_points = self.points[random_indices] + quaternion_apply(
            self.quaternions[random_indices], 
            sampling_scale_factor * self.scaling[random_indices] * torch.randn_like(self.points[random_indices]))
        
        return random_points, random_indices
    
    def drop_low_opacity_points(self, opacity_threshold=0.5):
        mask = self.strengths[..., 0] > opacity_threshold  # 1e-3, 0.5
        self.prune_points(mask)
    
    
    pass
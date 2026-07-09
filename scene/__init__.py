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

import os
import random
import json
from turtle import speed
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import open3d as o3d
from utils.gs_utils import save_mst_ply
import torch
from scene.models import CNN_decoder
from utils.gs_utils import build_mst_from_endpoints, save_mst_ply
from utils.general_utils import safe_state, get_expon_lr_func, build_rotation
from utils.visualization import TorchPCA 
import numpy as np
class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.args = args
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.pca = None
        self.pca_low = None
        if args.mask_path is not None:
            mask = args.mask_path
        else:
            mask = None
        if args.feature_path is not None:
            feature = args.feature_path
        else:
            feature = None

        args.pretrain_path =  "feature_pretrain" 
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, mask,feature, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)
        
        # --- PCA basis (dinov3_pca.pth): optional HERE, but required for PREPROCESSING ---
        # Step-2 training never *applies* it (semantic features arrive already reduced to
        # 128-d), so the load is optional and we skip gracefully if absent. IMPORTANT: the
        # basis is NOT disposable — the shipped text features (text_feats_dim128) were
        # projected with this exact 768->128 PCA, so any new scene's DINOv3 features must be
        # projected with the SAME basis in Step-1a to stay aligned with the text features.
        # It therefore has to be released alongside the dataset (see README Step 1a).
        pca_path = os.path.join(args.root_path, 'dinov3_pca.pth')
        if os.path.exists(pca_path):
            pca_checkpoint = torch.load(pca_path, map_location='cpu')
            gaussians.pca = pca_checkpoint['pca']
            gaussians.pca_low = pca_checkpoint['pca128']

        # --- text features (branch/leaf/bg prompts, already 128-d): REQUIRED ---
        # Ships in assets/ so the code works out of the box; prefer a copy at root_path.
        tf_path = os.path.join(args.root_path, "dinov3_text_feats.pth")
        if not os.path.exists(tf_path):
            tf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "assets", "dinov3_text_feats.pth")
        text_checkpoints = torch.load(tf_path, map_location='cpu')
        gaussians.text_feats = text_checkpoints['text_feats_dim128'][:3,].to(args.data_device)  # [stem,leaf]
        gaussians.text_feats_fgbg = text_checkpoints['text_feats_dim128'][2:,].to(args.data_device)  # [background, plant]
        
        if self.loaded_iter:
            # keep first two / in self.model_path 
            # base_path = "/".join(self.model_path.split("/")[:2])
            try:
                self.gaussians.load_ply(os.path.join(args.source_path,
                                                           f"{args.pretrain_path}/point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply")) # args.train_test # clean
            except: 
                self.gaussians.load_ply(os.path.join(args.source_path,
                                                            f"{args.pretrain_path}/point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "point_cloud.ply")) # args.train_test
            branch_points_path = os.path.join(args.source_path, "branch.ply")
            try:
                branch_gt = o3d.io.read_point_cloud(branch_points_path)
                self.gt_branch_points = torch.from_numpy(np.asarray(branch_gt.points)).float()
            except:
                self.gt_branch_points = None

        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, scene_info.semantic_feature_dim, speedup=args.speedup) # 384
    
    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        if self.gaussians.strprs is not None:
            self.gaussians.strprs.save_ply(os.path.join(point_cloud_path, "strpr.ply"))
            branch_mask, leaf_mask, _ = self.gaussians.get_cls_mask(label_threshold=0.5, mode='geo')
            self.gaussians.strprs.save_branch_ply(os.path.join(point_cloud_path, "strpr_branch.ply"), branch_mask)

            mst_edges, points, appgas_branch_points = self.gaussians.build_branch_graph()
            save_mst_ply( points, mst_edges, os.path.join(point_cloud_path, "mst.ply"))
            branch_pd = o3d.geometry.PointCloud()
            branch_pd.points = o3d.utility.Vector3dVector(appgas_branch_points.detach().cpu().numpy())
            o3d.io.write_point_cloud(os.path.join(point_cloud_path, "branch.ply"), branch_pd)
        if self.gaussians.appgas is not None and not os.path.exists(os.path.join(point_cloud_path, "appgas.ply")):
            self.gaussians.appgas.save_ply(os.path.join(point_cloud_path, "appgas.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_dynamic, render_edit
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.pytorch3d_compat import knn_points
import imageio
from torchvision.transforms.functional import to_pil_image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sklearn.decomposition
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
from scene.models import CNN_decoder
import yaml
from utils.graphics_utils import getWorld2View2
import cv2
from utils.pose_utils import render_path_spiral
from utils.visualization import feature_visualize_saving
import sys

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
    
    
def save_similarity_outputs(per_patch_similarity_to_text, save_path,prefix,image_pil, mask=None):
    pred_idx = per_patch_similarity_to_text.argmax(1).squeeze(0)
    probs = torch.softmax(per_patch_similarity_to_text, dim=1)   
    # save the predicted index map and similarity map
    pred_idx = probs.argmax(1).squeeze(0).cpu().numpy()  # H,W
    mask_np = mask.squeeze().detach().cpu().numpy()
    if mask_np.ndim == 3:  # (H,W) or (1,H,W)
        mask_np = mask_np.squeeze()
    if mask_np.shape != pred_idx.shape:
        mask_t = torch.from_numpy(mask_np).float()[None, None]  # (1,1,H,W)
        mask_t = F.interpolate(mask_t, size=pred_idx.shape, mode="nearest")
        mask_np = mask_t.squeeze().numpy()
    mask_bin = (mask_np > 0.5).astype(np.uint8)   # 前景=1 背景=0
    palette = np.array([
        [31, 119, 180],   # class 0: 蓝 (leaf)
        [44, 160, 44],    # class 1: 绿 (stem)
        [214, 39, 40],    # class 2: 红
        [148, 103, 189],  # class 3: 紫
        [255, 127, 14],   # class 4: 橙
    ], dtype=np.uint8)

    color_pred = palette[pred_idx]  # (H,W,3), uint8

    bg_color = np.array([0, 0, 0], dtype=np.uint8)
    out_rgb = np.where(mask_bin[..., None] == 1, color_pred, bg_color[None, None, :])

    # 如需与原图同尺寸：
    out_img = Image.fromarray(out_rgb, mode="RGB").resize(image_pil.size, Image.NEAREST)
    out_img.save(os.path.join(save_path,f"{prefix}_argmax_map.png"))    
    
def get_dino_text_feats(objects,scene):
    # load dinov3
    REPO_DIR = os.path.dirname(os.path.abspath(__file__))
    vitl16_backbone_path = "data/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    dino_txt_path = "data/checkpoints/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
    # dinov3, tokenizer = torch.hub.load(REPO_DIR, 'dinov3_vitl16_dinotxt_tet1280d20h24l', source='local', dinotxt_weights=dino_txt_path, backbone_weights=vitl16_backbone_path)
    # token = tokenizer.tokenize(objects)
    pca = scene.pca
    pca_low = scene.pca_low
    text_feats = scene.text_feats
    # text_features_aligned_to_patch = dinov3.encode_text(token)[:, 1024:] 
    # text_feat_pca = (text_features_aligned_to_patch - pca.mean_.cuda()) @ pca.components_.T.cuda()
    # text_feats = F.normalize(text_feat_pca.float(), p=2, dim=1)
    # text_feat_low = (text_feats - pca_low.mean_.cuda()) @ pca_low.components_.T.cuda()
    # text_feats = F.normalize(text_feat_low.float(), p=2, dim=1)
    
    return text_feats

def parse_edit_config_and_text_encoding(edit_config, gaussians):
    edit_dict = {}
    if edit_config is not None:
        with open(edit_config, 'r') as f:
            edit_config = yaml.safe_load(f)
            print(edit_config)
        objects = edit_config["edit"]["objects"]
        targets = edit_config["edit"]["targets"].split(",")
        edit_dict["positive_ids"] = [objects.index(t) for t in targets if t in objects]
        edit_dict["score_threshold"] = edit_config["edit"]["threshold"]
        
        # text encoding
        # text_feature = get_dino_text_feats(objects, scene)
        text_feature = gaussians.text_feats
        text_feature_fgbg = gaussians.text_feats_fgbg

        # setup editing
        op_dict = {}
        for operation in edit_config["edit"]["operations"].split(","):
            if operation == "extraction":
                op_dict["extraction"] = True
            elif operation == "deletion":
                op_dict["deletion"] = True
            elif operation == "color_func":
                op_dict["color_func"] = eval(edit_config["edit"]["colorFunc"])
            else:
                raise NotImplementedError
        edit_dict["operations"] = op_dict

        idx = edit_dict["positive_ids"][0]

    return edit_dict, text_feature, text_feature_fgbg,targets[idx]



def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, novel_view : bool, 
                video : bool , edit_config: str, novel_video : bool, multi_interpolate : bool, num_views : int): 
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, device=args.device)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, edit_config, scene,dataset.speedup)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, edit_config, scene,dataset.speedup)

        if novel_view:
             render_novel_views(dataset.model_path, "novel_views", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, 
                                edit_config, dataset.speedup, multi_interpolate, num_views)

        if video:
             render_video(dataset.model_path, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, edit_config)

        if novel_video:
             render_novel_video(dataset.model_path, "novel_views_video", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, edit_config, dataset.speedup)

def render_ours(model_path, name, iteration, views, gaussians, stprs,pipeline, background, train_test_exp, separate_sh):
    # for deformable appgs rendering
    render_path = os.path.join(model_path,  'renders',"ours_{}".format(iteration))
    os.makedirs(render_path, exist_ok=True)
    gts_path = os.path.join(model_path,  "ours_{}".format(iteration), "gt")
    stprs_path = os.path.join(model_path,  "point_cloud/iteration_{}".format(iteration), "stprs.ply")
    stprs.load_ply(stprs_path)
    makedirs(render_path, exist_ok=True)
    # makedirs(gts_path, exist_ok=True)
    stprs_xyz = stprs.get_xyz
    app_xyz = gaussians.get_xyz
    knn_idx = knn_points(app_xyz.unsqueeze(0), stprs_xyz.unsqueeze(0), K=1, return_nn=True)[1][0] 
    stpr_label = stprs.label
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering_list = render_dynamic(view, gaussians, pipeline, knn_idx,background, stpr_label,use_trained_exp=train_test_exp, separate_sh=separate_sh)
        gt = view.original_image[0:3, :, :]
        # save list to gif 
        # rendering = torch.stack(rendering_list, dim=0)
        frames = []
        for img in rendering_list:
            if img.is_cuda:
                img = img.cpu()
            pil = to_pil_image(img)
            frames.append(pil)
        imageio.mimsave(os.path.join(render_path, '{0:05d}'.format(idx) + ".gif"), frames, duration=0.1)
        print("Rendering " + os.path.join(render_path, '{0:05d}'.format(idx) + ".gif"))
        break

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, edit_config, scene,speedup):
    if edit_config != "no editing":
        edit_dict, text_feature, text_feature_fgbg,target = parse_edit_config_and_text_encoding(edit_config, gaussians)

        edit_render_path = os.path.join(model_path, name, "ours_{}_{}_{}".format(iteration, next(iter(edit_dict["operations"])), target), "renders")
        edit_gts_path = os.path.join(model_path, name, "ours_{}_{}_{}".format(iteration, next(iter(edit_dict["operations"])), target), "gt")
        edit_feature_map_path = os.path.join(model_path, name, "ours_{}_{}_{}".format(iteration, next(iter(edit_dict["operations"])), target), "feature_map")
        edit_gt_feature_map_path = os.path.join(model_path, name, "ours_{}_{}_{}".format(iteration, next(iter(edit_dict["operations"])), target), "gt_feature_map")

        makedirs(edit_render_path, exist_ok=True)
        makedirs(edit_gts_path, exist_ok=True)
        makedirs(edit_feature_map_path, exist_ok=True)
        makedirs(edit_gt_feature_map_path, exist_ok=True)
    
    else:
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
        feature_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_map")
        gt_feature_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_feature_map")
        saved_feature_path = os.path.join(model_path, name, "ours_{}".format(iteration), "saved_feature")
        #encoder_ckpt_path = os.path.join(model_path, "encoder_chkpnt{}.pth".format(iteration))
        decoder_ckpt_path = os.path.join(model_path, "decoder_chkpnt{}.pth".format(iteration))
        depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth") ###
        
        if speedup:
            gt_feature_map = views[0].semantic_feature.cuda()
            feature_out_dim = gt_feature_map.shape[0]
            feature_in_dim = int(feature_out_dim/4)
            cnn_decoder = CNN_decoder(feature_in_dim, feature_out_dim)
            cnn_decoder.load_state_dict(torch.load(decoder_ckpt_path))
        
        makedirs(render_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)
        makedirs(feature_map_path, exist_ok=True)
        makedirs(gt_feature_map_path, exist_ok=True)
        makedirs(saved_feature_path, exist_ok=True)
        makedirs(depth_path, exist_ok=True) ###

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if edit_config != "no editing":
            render_pkg = render_edit(view, gaussians, pipeline, background, text_feature, edit_dict) 
            gt = view.original_image[0:3, :, :]
            gt_feature_map = view.semantic_feature.cuda() 
            torchvision.utils.save_image(render_pkg["render"], os.path.join(edit_render_path, '{0:05d}'.format(idx) + ".png")) 

            # torchvision.utils.save_image(render_pkg["render_leaf"], os.path.join(edit_render_path, '{0:05d}'.format(idx) + "_leaf.png")) 
            # torchvision.utils.save_image(render_pkg["render_branch"], os.path.join(edit_render_path, '{0:05d}'.format(idx) + "_branch.png")) 

            torchvision.utils.save_image(gt, os.path.join(edit_gts_path, '{0:05d}'.format(idx) + ".png"))
            # visualize feature map
            feature_map = render_pkg["feature_map"]
            feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0) ###
            if speedup:
                feature_map = cnn_decoder(feature_map)

            feature_map_vis = feature_visualize_saving(feature_map)
            Image.fromarray((feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(edit_feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))
            gt_feature_map_vis = feature_visualize_saving(gt_feature_map)
            Image.fromarray((gt_feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(edit_gt_feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))
            # # test similarity map saving
            # text_feature = F.normalize(text_feature, p=2, dim=1)  # C_text
            # feature_map = F.normalize(feature_map, p=2, dim=0)
            # per_patch_similarity_to_text = torch.einsum("bdhw,cd->bchw", feature_map.unsqueeze(0), text_feature)
            # feature_path = model_path
            # data_path = 'data/plant0'
            # image = Image.open(os.path.join(data_path, "images/IMG_1388.JPG")).convert("RGB")
            # image_name = "IMG_1388"
            # mask = Image.open(os.path.join(data_path, "masks/IMG_1388.JPG")).convert("L")
            # img_size=448
        
            # transform_mask = T.Compose(
            #     [
            #         T.Resize(img_size, interpolation=T.InterpolationMode.NEAREST),
            #         T.CenterCrop(img_size),
            #         T.ToTensor(),  # Convert to tensor
            #     ]
            # )
            # mask_batch = transform_mask(mask).unsqueeze(0)
            # save_similarity_outputs(per_patch_similarity_to_text, save_path=feature_path,
            #                         prefix=image_name, image_pil = image, mask=mask_batch)    

        else:
            render_pkg = render(view, gaussians, pipeline, background) 

            gt = view.original_image[0:3, :, :]
            gt_feature_map = view.semantic_feature.cuda() 
            torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png")) 
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            
            ### depth ###
            depth = render_pkg["depth"]
            scale_nor = depth.max().item()
            depth_nor = depth / scale_nor
            depth_tensor_squeezed = depth_nor.squeeze()  # Remove the channel dimension
            colormap = plt.get_cmap('jet')
            depth_colored = colormap(depth_tensor_squeezed.cpu().numpy())
            depth_colored_rgb = depth_colored[:, :, :3]
            depth_image = Image.fromarray((depth_colored_rgb * 255).astype(np.uint8))
            output_path = os.path.join(depth_path, '{0:05d}'.format(idx) + ".png")
            depth_image.save(output_path)
            ##############

            # visualize feature map
            feature_map = render_pkg["feature_map"] 
            feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0) ###
            if speedup:
                feature_map = cnn_decoder(feature_map)

            feature_map_vis = feature_visualize_saving(feature_map)
            Image.fromarray((feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))
            gt_feature_map_vis = feature_visualize_saving(gt_feature_map)
            Image.fromarray((gt_feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(gt_feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))

            # save feature map
            feature_map = feature_map.cpu().numpy().astype(np.float16)
            torch.save(torch.tensor(feature_map).half(), os.path.join(saved_feature_path, '{0:05d}'.format(idx) + "_fmap_CxHxW.pt"))

def render_novel_views(model_path, name, iteration, views, gaussians, pipeline, background, 
                       edit_config, speedup, multi_interpolate, num_views):
    if multi_interpolate:
        name = name + "_multi_interpolate"
    # make dirs
    if edit_config != "no editing":
        edit_dict, text_feature, target = parse_edit_config_and_text_encoding(edit_config)
        
        # edit
        edit_render_path = os.path.join(model_path, name, "ours_{}_{}_{}".format(iteration, next(iter(edit_dict["operations"])), target), "renders")
        edit_feature_map_path = os.path.join(model_path, name, "ours_{}_{}_{}".format(iteration, next(iter(edit_dict["operations"])), target), "feature_map")

        makedirs(edit_render_path, exist_ok=True)
        makedirs(edit_feature_map_path, exist_ok=True)
    else:
        # non-edit
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
        feature_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_map")
        saved_feature_path = os.path.join(model_path, name, "ours_{}".format(iteration), "saved_feature")
        #encoder_ckpt_path = os.path.join(model_path, "encoder_chkpnt{}.pth".format(iteration))
        decoder_ckpt_path = os.path.join(model_path, "decoder_chkpnt{}.pth".format(iteration))

        if speedup:
            gt_feature_map = views[0].semantic_feature.cuda()
            feature_out_dim = gt_feature_map.shape[0]
            feature_in_dim = int(feature_out_dim/4)
            cnn_decoder = CNN_decoder(feature_in_dim, feature_out_dim)
            cnn_decoder.load_state_dict(torch.load(decoder_ckpt_path))
        
        makedirs(render_path, exist_ok=True)
        makedirs(feature_map_path, exist_ok=True)
        makedirs(saved_feature_path, exist_ok=True)

    view = views[0]
    
    # create novel poses
    render_poses = []
    for cam in views:
        pose = np.concatenate([cam.R, cam.T.reshape(3, 1)], 1)
        render_poses.append(pose) 
    if not multi_interpolate:
        poses = interpolate_matrices(render_poses[0], render_poses[-1], num_views)
    else:
        poses = multi_interpolate_matrices(np.array(render_poses), 2)

    # rendering process
    for idx, pose in enumerate(tqdm(poses, desc="Rendering progress")):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:, :3], pose[:, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]

        if edit_config != "no editing":
            render_pkg = render_edit(view, gaussians, pipeline, background, text_feature, edit_dict)
            gt = view.original_image[0:3, :, :]
            gt_feature_map = view.semantic_feature.cuda()
            torchvision.utils.save_image(render_pkg["render"], os.path.join(edit_render_path, '{0:05d}'.format(idx) + ".png")) 
            # visualize feature map
            feature_map = render_pkg["feature_map"] 
            feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0) ###
            if speedup:
                feature_map = cnn_decoder(feature_map)

            feature_map_vis = feature_visualize_saving(feature_map)
            Image.fromarray((feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(edit_feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))
        else:
            # mlp encoder
            render_pkg = render(view, gaussians, pipeline, background) 

            gt_feature_map = view.semantic_feature.cuda() 
            torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png")) 
            # visualize feature map
            feature_map = render_pkg["feature_map"]
            feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0) ###
            if speedup:
                feature_map = cnn_decoder(feature_map)

            feature_map_vis = feature_visualize_saving(feature_map)
            Image.fromarray((feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))

            # save feature map
            feature_map = feature_map.cpu().numpy().astype(np.float16)
            torch.save(torch.tensor(feature_map).half(), os.path.join(saved_feature_path, '{0:05d}'.format(idx) + "_fmap_CxHxW.pt"))

def render_video(model_path, iteration, views, gaussians, pipeline, background, edit_config): ###
    render_path = os.path.join(model_path, 'video', "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    view = views[0]
    render_poses = render_path_spiral(views)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (view.original_image.shape[2], view.original_image.shape[1])
    final_video = cv2.VideoWriter(os.path.join(render_path, 'final_video.mp4'), fourcc, 10, size)

    if edit_config != "no editing":
        edit_dict, text_feature = parse_edit_config_and_text_encoding(edit_config)

    for idx, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]

        if edit_config != "no editing":
            rendering = torch.clamp(render_edit(view, gaussians, pipeline, background, text_feature, edit_dict)["render"], min=0., max=1.) ###
        else:
            rendering = torch.clamp(render(view, gaussians, pipeline, background)["render"], min=0., max=1.)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        final_video.write((rendering.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1])
    final_video.release()



def interpolate_matrices(start_matrix, end_matrix, steps):
        # Generate interpolation factors
        interpolation_factors = np.linspace(0, 1, steps)
        # Interpolate between the matrices
        interpolated_matrices = []
        for factor in interpolation_factors:
            interpolated_matrix = (1 - factor) * start_matrix + factor * end_matrix
            interpolated_matrices.append(interpolated_matrix)
        return np.array(interpolated_matrices)


def multi_interpolate_matrices(matrix, num_interpolations):
    interpolated_matrices = []
    for i in range(matrix.shape[0] - 1):
        start_matrix = matrix[i]
        end_matrix = matrix[i + 1]
        for j in range(num_interpolations):
            t = (j + 1) / (num_interpolations + 1)
            interpolated_matrix = (1 - t) * start_matrix + t * end_matrix
            interpolated_matrices.append(interpolated_matrix)
    return np.array(interpolated_matrices)

def render_novel_video(model_path, name, iteration, views, gaussians, pipeline, background, edit_config): 
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    view = views[0]
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (view.original_image.shape[2], view.original_image.shape[1])
    final_video = cv2.VideoWriter(os.path.join(render_path, 'final_video.mp4'), fourcc, 10, size)

    if edit_config != "no editing":
        edit_dict, text_feature = parse_edit_config_and_text_encoding(edit_config)
    
    render_poses = [(cam.R, cam.T) for cam in views]
    render_poses = []
    for cam in views:
        pose = np.concatenate([cam.R, cam.T.reshape(3, 1)], 1)
        render_poses.append(pose)
    
    # create novel poses
    poses = interpolate_matrices(render_poses[0], render_poses[-1], 200) 

    # rendering process
    for idx, pose in enumerate(tqdm(poses, desc="Rendering progress")):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:, :3], pose[:, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]

        if edit_config != "no editing":
            rendering = torch.clamp(render_edit(view, gaussians, pipeline, background, text_feature, edit_dict)["render"], min=0., max=1.) 
        else:
            rendering = torch.clamp(render(view, gaussians, pipeline, background)["render"], min=0., max=1.)
        
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        final_video.write((rendering.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1])
    final_video.release()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--novel_view", action="store_true") ###
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--video", action="store_true") ###
    parser.add_argument("--novel_video", action="store_true") ###
    parser.add_argument('--edit_config', default="no editing", type=str)
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')

    parser.add_argument("--multi_interpolate", action="store_true") ###
    parser.add_argument("--num_views", default=200, type=int)
    args = get_combined_args(parser)
    # args = parser.parse_args(sys.argv[1:])

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    print(f"Using device {device}")
    args.device = device
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.novel_view, 
                args.video, args.edit_config, args.novel_video, args.multi_interpolate, args.num_views)
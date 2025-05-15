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
from gaussian_renderer import render, render_dynamic
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from pytorch3d.ops import knn_points
import imageio
from torchvision.transforms.functional import to_pil_image

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        rendering,_,_ = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        rendering = rendering['render']
        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, render_deform: bool,skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree,device="cuda")
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        stprs = GaussianModel(dataset.sh_degree, device="cuda")

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

       
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if render_deform:
            render_ours(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, stprs,pipeline, background, dataset.train_test_exp, separate_sh)


        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

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



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--render_deform", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.render_deform,args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)
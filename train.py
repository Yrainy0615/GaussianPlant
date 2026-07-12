#
# GaussianPlant training: jointly optimise structure primitives (StrPr) and
# appearance gaussians (AppGS), built from a pre-trained feature 3DGS.
#
import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import torch
import torch.nn.functional as F
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render, render_gsplant
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim, laplacian_smooth_loss, loss_endpoints, label_loss

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, debug_from, args):
    tb_writer = prepare_output_and_logger(dataset)

    # Load the pre-trained feature 3DGS, then derive StrPr (clusters) and AppGS.
    gaussians_pretrain = GaussianModel(dataset.sh_degree, args.device, args.model_path)
    scene = Scene(dataset, gaussians_pretrain, load_iteration=checkpoint)

    # Background removal is essential so that colour can cleanly classify branch (brown)
    # vs leaf (green): the raw feature cloud is ~86% pot/background. Prefer the dataset's
    # pre-cleaned point cloud (pretrain_clean/*_clean_pruned.ply); otherwise fall back to
    # masking via the 2D plant masks.
    if args.clean_ply:
        clean_path = args.clean_ply if os.path.isabs(args.clean_ply) else os.path.join(args.root_path, args.clean_ply)
        print(f"Loading background-removed point cloud: {clean_path}")
        gaussians_pretrain.load_ply(clean_path)
    elif args.rm_bg:
        plant_mask = gaussians_pretrain.compute_plant_mask(scene.getTrainCameras())
        gaussians_pretrain.apply_mask(plant_mask)

    strpr = gaussians_pretrain.build_strpr_from_gs(
        cluster_method="kmeans", vis_cluster=False, max_strpr_num=args.max_strpr_num,
        label_init=args.label_init, branch_frac=args.branch_frac, cluster_size=args.cluster_size)
    strpr.training_setup(opt)

    appgas = gaussians_pretrain.build_appgas_from_stprs(use_pretrain=True)
    appgas.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final,
                                        max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    progress_bar = tqdm(range(1, opt.iterations + 1), desc="Training progress")

    for iteration in range(1, opt.iterations + 1):
        iter_start.record()
        strpr.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            strpr.oneupSHdegree()

        # Pick a random camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render StrPr (branch-isolated) and AppGS
        render_pkg_strpr, _, render_branch_strpr = render_gsplant(
            viewpoint_cam, strpr, pipe, background, render_only_branch=True)
        render_strpr = render_pkg_strpr["render"]
        feature_map_strpr = render_pkg_strpr["feature_map"]
        viewspace_point_tensor = render_pkg_strpr["viewspace_points"]
        visibility_filter = render_pkg_strpr["visibility_filter"]
        radii = render_pkg_strpr["radii"]

        render_pkg_appgas, _, _ = render_gsplant(viewpoint_cam, appgas, pipe, background)
        render_appgas = render_pkg_appgas["render"]
        feature_map = render_pkg_appgas["feature_map"]
        radii_app = render_pkg_appgas["radii"]
        viewspace_point_tensor_app = render_pkg_appgas["viewspace_points"]
        visibility_filter_app = render_pkg_appgas["visibility_filter"]

        # Ground truth for this view
        gt_image = viewpoint_cam.original_image.cuda()
        mask = viewpoint_cam.mask.cuda() if viewpoint_cam.mask is not None else 1.0
        gt_feature_map = viewpoint_cam.semantic_feature.cuda() if viewpoint_cam.semantic_feature is not None else None

        # --- Photometric: StrPr should track AppGS, AppGS should track the image ---
        # AppGS is the TEACHER (detached): StrPr (sparse, ~1.6k) chases the AppGS render, but the
        # gradient must NOT flow back into AppGS. Previously render_appgas was NOT detached, so in
        # plant pixels the sparse StrPr leaves uncovered (black holes) the L1 dragged AppGS toward
        # black to match StrPr -> AppGS rendered ~4.5x too dark with holes, fighting its own GT term.
        # Also mask the term so only the plant region is matched (background is black in both).
        Ll1 = l1_loss(render_strpr * mask, render_appgas.detach() * mask)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + \
            opt.lambda_dssim * (1.0 - ssim(render_strpr * mask, gt_image * mask))

        Ll1_appgas = l1_loss(render_appgas * mask, gt_image * mask)
        loss += (1.0 - opt.lambda_dssim) * Ll1_appgas + \
            opt.lambda_dssim * (1.0 - ssim(render_appgas * mask, gt_image * mask))

        # --- Semantic feature supervision (DINOv3) ---
        Ll1_feature = 0.0
        if args.reg_sem and gt_feature_map is not None and feature_map is not None:
            gt_feature_map = F.interpolate(
                gt_feature_map.unsqueeze(0), size=(feature_map.shape[1], feature_map.shape[2]),
                mode='bilinear', align_corners=True).squeeze(0)
            feat_loss = l1_loss(feature_map * mask, gt_feature_map * mask)
            loss += feat_loss
            Ll1_feature = feat_loss.item()

        # --- Depth regularisation (mono inv-depth; only when a reliable map exists) ---
        Ll1depth = 0.0
        if args.reg_depth and depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg_strpr["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
            loss += depth_l1_weight(iteration) * Ll1depth_pure
            Ll1depth = Ll1depth_pure.item()

        # --- Binding (StrPr surface fit to AppGS) + semantic label refinement ---
        # compute_for_appgas=False keeps AppGS geometry fixed (the cleaned plant points are
        # already good); binding only moves the StrPr cylinders/disks to fit them and refines
        # the branch/leaf label, instead of dragging AppGS off the skeleton.
        if args.reg_bind and iteration > args.densify_from_iter:
            bind_loss, dis_cylinder, dis_disk = gaussians_pretrain.compute_binding_loss(
                compute_for_appgas=args.bind_move_appgas, prior='geo',
                detach_label=not args.bind_label)
            reg_semantic = gaussians_pretrain.semantic_loss()
        else:
            bind_loss = torch.tensor(0.0, device=args.device)
            dis_cylinder = dis_disk = 0
            reg_semantic = torch.tensor(0.0, device=args.device)

        # --- Overlap regulariser: REMOVED ---
        # The SuGaR-style overlap loss minimised neighbour overlap by collapsing each Gaussian's
        # cross-section, which streaked the LEAF StrPr into thin needles (in-plane elongation
        # 1.8 -> ~185 on some scenes) while leaving the branch Chamfer unchanged. Dropped from the
        # pipeline; loss_op is kept as a constant 0 only so the tensorboard logging signature is
        # unchanged. (compute_gaussian_overlap_with_neighbors stays in the model, just unused.)
        loss_op = torch.tensor(0.0, device="cuda")

        # --- Graph (tree continuity / smoothness over the branch MST) ---
        loss_graph = torch.tensor(0.0, device="cuda")
        if args.reg_graph and iteration > args.graph_from and iteration % args.graph_interval == 0:
            mst_edges, points, _ = gaussians_pretrain.build_branch_graph()
            if points.shape[0] > 0:
                loss_graph = laplacian_smooth_loss(points, mst_edges) + \
                    loss_endpoints(points[0::2], points[1::2], mst_edges)

        # --- Label optimisation: geometric-binding DIMENSIONALITY cue + colour, anti-collapse ---
        # The branch/leaf label is driven by the binding geometry read in a size-INVARIANT way:
        # the dimensionality of each StrPr's bound AppGS (linearity=1D branch vs planarity=2D leaf),
        # combined with the absolute colour cue (brown=branch). A branch-fraction prior keeps the
        # minority branch class alive. This replaces the cylinder/disk *distance* difference (which
        # collapsed: a wide disk captures everything -> all leaf) and the symmetric p(1-p) loss.
        if args.reg_cls:
            s_shape, valid = gaussians_pretrain.compute_shape_prior()          # [-1,1], high->branch
            rgb = 0.2820948 * strpr._features_dc.squeeze(1) + 0.5
            greenness = rgb[:, 1] - 0.5 * (rgb[:, 0] + rgb[:, 2])
            s_col = -(greenness - greenness.mean()) / (greenness.std() + 1e-6)  # high->brown->branch
            target = torch.sigmoid(opt.w_shape * s_shape + opt.w_col_lab * s_col)
            p = strpr.label_activation(strpr.label).squeeze()
            if valid.any():
                loss += opt.lambda_label * ((p[valid] - target[valid].detach()) ** 2).mean()
            loss += opt.lambda_bfrac * (p.mean() - args.branch_frac) ** 2

        # --- Alignment (neighbouring StrPr orientation/scale, SuGaR-style spacing) ---
        if args.reg_align:
            strpr.reset_neighbors()
            loss += opt.lambda_align * strpr.compute_gaussian_alignment_with_neighbors(strpr.knn_idx).mean()

        # --- Branch axis alignment: orient each branch StrPr's long axis ALONG the branch ---
        loss_axis = torch.tensor(0.0, device="cuda")
        if args.reg_axis and iteration > args.densify_from_iter:
            loss_axis = gaussians_pretrain.compute_branch_axis_alignment()

        total_loss = (loss + opt.lambda_bind * bind_loss
                      + opt.lambda_axis * loss_axis
                      + opt.lambda_graph * loss_graph + opt.lambda_sem * reg_semantic)
        total_loss.backward()

        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{Ll1.item():.5f}",
                                          "StrPr num": f"{len(strpr.get_xyz)}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                            iter_start.elapsed_time(iter_end), testing_iterations, scene,
                            render, (pipe, background), dataset.train_test_exp,
                            render_strpr, feature_map, render_appgas, gt_feature_map,
                            feature_map_strpr, render_branch_strpr,
                            dis_cylinder, dis_disk, loss_op, loss_graph,
                            Ll1_feature, 0.0, reg_semantic.item())

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification. AppGS are the cleaned plant points and stay fixed by default
            # (--densify_appgas to enable); only StrPr densify, capped at max_strpr_num.
            if iteration < opt.densify_until_iter:
                strpr.max_radii2D[visibility_filter] = torch.max(
                    strpr.max_radii2D[visibility_filter], radii[visibility_filter])
                strpr.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if args.densify_appgas:
                    appgas.max_radii2D[visibility_filter_app] = torch.max(
                        appgas.max_radii2D[visibility_filter_app], radii_app[visibility_filter_app])
                    appgas.add_densification_stats(viewspace_point_tensor_app, visibility_filter_app)

                if args.densify and iteration > opt.densify_from_iter and \
                        iteration % opt.densification_interval == 0:
                    size_threshold = 400
                    # gsplat means2d gradients are ~1e3x smaller than the original CUDA rasteriser's
                    # (NDC-normalised), so the original threshold (~2e-4) never fires. Use an adaptive
                    # threshold = a high quantile of the current accumulated gradients, so we densify
                    # the most-under-reconstructed top fraction regardless of rasteriser scale; it
                    # naturally tapers as gradients fall.
                    if strpr.get_xyz.shape[0] < args.max_strpr_num:
                        gg = (strpr.xyz_gradient_accum / strpr.denom.clamp_min(1)).norm(dim=-1)
                        thr = torch.quantile(gg[gg > 0], args.densify_quantile).item() if (gg > 0).any() else 1e9
                        strpr.densify_and_prune(thr, 0.01, scene.cameras_extent, size_threshold, radii)
                    else:
                        strpr.densify_and_prune(opt.densify_grad_threshold, 0.01,
                                                scene.cameras_extent, size_threshold, radii, only_prune=True)
                    if args.densify_appgas:
                        appgas.densify_and_prune(opt.densify_grad_threshold, 0.005,
                                                 scene.cameras_extent, 20, None, radii_app)

            # Periodic structural pruning: demote floating/isolated false-branch StrPrs to leaf
            # (branches form a connected tree; mis-classified leaves float). The strongest lever
            # for the branch Chamfer (removes the false branches that dominate pred->gt error).
            if args.prune_isolated and args.prune_from <= iteration <= args.prune_until \
                    and iteration % args.prune_interval == 0:
                nd = gaussians_pretrain.prune_isolated_branches(
                    iso_factor=args.prune_iso_factor,
                    len_weight=args.prune_len_weight, turn_weight=args.prune_turn_weight)
                msg = f"\n[ITER {iteration}] pruned {nd} isolated branch StrPr -> leaf"
                if args.prune_green:  # also remove clearly-green floating StrPr (visual cleanup)
                    ng = gaussians_pretrain.prune_green_floats(green_z=args.prune_green_z)
                    msg += f", {ng} green floats"
                print(msg)

            # World-scale prune: delete degenerate low-frequency StrPr whose world max scale
            # exceeds prune_scale_frac * scene extent (the 'ambient wash' blobs centred inside the
            # plant that the 2D mask cannot exclude). Geometry-based, so it complements the
            # photometric/opacity prunes. NOT capped at prune_until (unlike the branch-structure
            # prunes): the blobs regrow whenever left unpruned, so this runs through to the end.
            if args.prune_large_scale and iteration >= args.prune_from \
                    and iteration <= opt.iterations - 100 \
                    and iteration % args.prune_interval == 0:
                nb = gaussians_pretrain.prune_large_scale(
                    scene.cameras_extent, scale_frac=args.prune_scale_frac,
                    opacity_below=(args.prune_scale_opacity if args.prune_scale_opacity > 0 else None))
                if nb:
                    print(f"\n[ITER {iteration}] pruned {nb} oversized StrPr "
                          f"(> {args.prune_scale_frac:.0%} scene extent), now {strpr.get_xyz.shape[0]}")

            # Structure-driven BRANCH densification: split branch StrPr that under-cover the AppGS
            # they bind (along-axis spill), filling long branch segments with more cylinders. Uses
            # binding geometry, not photometric gradient, so it densifies the skeleton (not leaves).
            if args.densify_branch and args.prune_from <= iteration <= args.prune_until \
                    and iteration % args.prune_interval == 0:
                ns = gaussians_pretrain.densify_branches(
                    ratio_thr=args.branch_split_ratio, max_strpr_num=args.max_strpr_num)
                if ns:
                    print(f"\n[ITER {iteration}] split {ns} under-covering branch StrPr "
                          f"(now {strpr.get_xyz.shape[0]} StrPr)")

            # Optimiser step
            if iteration < opt.iterations:
                strpr.optimizer.step()
                strpr.optimizer.zero_grad(set_to_none=True)
                appgas.optimizer.step()
                appgas.optimizer.zero_grad(set_to_none=True)

            if iteration == checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((strpr.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = os.getenv('OAR_JOB_ID') or str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations,
                    scene: Scene, renderFunc, renderArgs, train_test_exp,
                    render_strpr, feature_map, render_appgas, gt_feature_map, feature_map_strpr,
                    render_branch, dis_cylinder, dis_disk, loss_op, loss_graph,
                    l1_feature, l1_feature_strpr, reg_semantic):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('train_loss_patches/loss_op', loss_op.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/loss_graph', loss_graph.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/dis_cylinder', dis_cylinder, iteration)
        tb_writer.add_scalar('train_loss_patches/dis_disk', dis_disk, iteration)
        tb_writer.add_scalar('train_loss_patches/num_strprs', scene.gaussians.strprs.get_xyz.shape[0], iteration)
        tb_writer.add_scalar('train_loss_patches/num_appgas', scene.gaussians.appgas.get_xyz.shape[0], iteration)
        tb_writer.add_scalar('train_loss_patches/l1_feature', l1_feature, iteration)
        tb_writer.add_scalar('train_loss_patches/reg_semantic', reg_semantic, iteration)
        if iteration % 50 == 0:
            tb_writer.add_images('render/strpr', render_strpr[None], iteration)
            tb_writer.add_images('render/appgas', render_appgas[None], iteration)
            if render_branch is not None:
                tb_writer.add_images('render/branch', render_branch[None], iteration)
        torch.cuda.empty_cache()

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras': scene.getTestCameras()},
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                                          for idx in range(5, 30, 5)]})
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians.appgas, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 1000, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int,
                        default=[100, 500, 1000, 3000, 5000, 6000, 7000, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", type=int, default=-1)
    parser.add_argument("--load_iteration", type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')
    parser.add_argument("--reg_graph", action="store_true", default=False)
    parser.add_argument("--graph_from", type=int, default=1000,
                        help="iteration after which the MST graph regulariser is active")
    parser.add_argument("--graph_interval", type=int, default=50,
                        help="recompute the (expensive) MST graph loss every N iters")
    parser.add_argument("--reg_align", action="store_true", default=False)
    parser.add_argument("--reg_axis", action="store_true", default=False,
                        help="align each branch StrPr's long axis along the local branch direction")
    parser.add_argument("--reg_bind", action="store_true", default=False)
    parser.add_argument("--prune_large_scale", action="store_true", default=False,
                        help="periodically delete StrPr whose world max-scale > prune_scale_frac * "
                             "scene extent (removes the low-frequency ambient-wash blobs the 2D mask "
                             "cannot catch)")
    parser.add_argument("--prune_scale_frac", type=float, default=0.2,
                        help="world-scale prune threshold as a fraction of the plant bounding box")
    parser.add_argument("--prune_scale_opacity", type=float, default=0.15,
                        help="only prune oversized StrPr whose opacity is below this, so real (opaque) "
                             "leaf disks are protected and only faint ambient-wash blobs are removed; "
                             "set 0 to prune purely by scale")
    parser.add_argument("--reg_cls", action="store_true", default=False,
                        help="colour-contrast + confidence loss to binarise branch/leaf labels")
    parser.add_argument("--reg_sem", action="store_true", default=False)
    parser.add_argument("--reg_depth", action="store_true", default=False)
    parser.add_argument("--densify", action="store_true", default=False)
    parser.add_argument("--densify_branch", action="store_true", default=False,
                        help="binding-driven branch-only densification: split branch StrPr whose "
                             "bound AppGS spill past the cylinder, filling long branch segments")
    parser.add_argument("--branch_split_ratio", type=float, default=1.5,
                        help="split a branch StrPr when bound-AppGS along-axis extent > ratio*length")
    parser.add_argument("--densify_quantile", type=float, default=0.9,
                        help="densify StrPr whose accumulated grad exceeds this quantile (gsplat "
                             "grads are ~1e3x smaller than CUDA-rasteriser, so use a quantile not a "
                             "fixed threshold); 0.9 -> top 10% split/cloned each interval")
    parser.add_argument("--densify_appgas", action="store_true", default=False,
                        help="also densify/prune AppGS (off by default: cleaned points stay fixed)")
    parser.add_argument("--bind_move_appgas", action="store_true", default=False,
                        help="let the binding loss move AppGS positions (off: only StrPr move)")
    parser.add_argument("--bind_label", action="store_true", default=False,
                        help="let the binding loss drive the branch/leaf label (off: binding "
                             "refines geometry only; colour drives the label)")
    parser.add_argument("--rm_bg", action="store_true", default=False,
                        help="remove pot/background Gaussians via 2D plant masks before init")
    parser.add_argument("--clean_ply", type=str, default="",
                        help="path to a pre-cleaned (background-removed) point cloud, e.g. "
                             "pretrain_clean/newplant1_clean_pruned.ply (relative to --root_path)")
    parser.add_argument("--max_strpr_num", type=int, default=400)
    parser.add_argument("--label_init", type=str, default="joint",
                        help='StrPr branch/leaf init: "joint" | "anisotropy" | "semantic"')
    parser.add_argument("--branch_frac", type=float, default=-1.0,
                        help="fraction of StrPr initialised as branch; <=0 auto-calibrates (Otsu, capped)")
    parser.add_argument("--cluster_size", type=int, default=100,
                        help="avg points per StrPr cluster (smaller -> finer/more StrPr)")
    parser.add_argument("--prune_isolated", action="store_true", default=False,
                        help="periodically demote spatially-isolated (floating) branch StrPr to leaf")
    parser.add_argument("--prune_iso_factor", type=float, default=2.5,
                        help="demote branch StrPr whose isolation > median*factor (adaptive; 2.5 generalises)")
    parser.add_argument("--prune_len_weight", type=float, default=0.0,
                        help="optional MST edge-length cue weight for pruning (0=off; isolation alone is best)")
    parser.add_argument("--prune_turn_weight", type=float, default=0.0,
                        help="optional MST turn-angle cue weight for pruning (0=off; weak/inconsistent)")
    parser.add_argument("--prune_green", action="store_true", default=False,
                        help="also demote clearly-green isolated 'floating' StrPr (visual cleanup; "
                             "may slightly raise Chamfer on clean scenes)")
    parser.add_argument("--prune_green_z", type=float, default=0.8,
                        help="greenness z-score threshold for green-float pruning")
    parser.add_argument("--prune_interval", type=int, default=500)
    parser.add_argument("--prune_from", type=int, default=1000)
    parser.add_argument("--prune_until", type=int, default=3000)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    torch.cuda.set_device(args.gpu)
    args.device = torch.device(f"cuda:{args.gpu}")
    print(f"Using device {args.device}")
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations,
             args.save_iterations, args.checkpoint_iterations, args.load_iteration, args.debug_from, args)
    print("\nTraining complete.")

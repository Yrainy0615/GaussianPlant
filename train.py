import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim,  align_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, save_tensor_as_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from pytorch3d.loss import chamfer_distance
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians_init = GaussianModel(dataset.sh_degree, opt.optimizer_type, args.device)
    appgs = None
    stprs = None
    scene = Scene(dataset, gaussians_init)
    gaussians_init.training_setup(opt)
    gt_xyz = gaussians_init.get_xyz.detach()
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians_init.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    process_state = "init"

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    # regularizer           
    loss_align = 0
    loss_overlap = 0
    loss_freq = 0
    num_appgs = 0
    num_stprs = 0
    loss_bind = 0
    loss_opacity_app = 0
    loss_opacity_stprs = 0
    image_stprs = None
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians_init, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        # build our representation
        if os.path.exists(os.path.join(args.source_path, "points_3dgs.ply")):
            opt.iter_build_stprs= 1
        else:
            opt.iter_build_stprs = 7001
        if iteration == opt.iter_build_stprs:   # build stprs from inital gaussians
            stprs = gaussians_init.build_stprs_from_gs(num_clusters=100, method='3dgs')
            stprs.training_setup(opt)
            stprs.save_ply(os.path.join(scene.model_path, "stprs_init.ply"))
            process_state = "stprs"
        if iteration == opt.iter_build_appgs :# build appgs from stprs
            cylinder_paras = stprs.convert_gs_to_cylinders()
            appgs = gaussians_init.build_appgs_from_cylinder(cylinder_paras,num_samples=50)
            print('binding appgs to stprs')
            appgs.training_setup(opt)
            process_state = "appgs"
 
         # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)
        # Load GT
        gt_image = viewpoint_cam.original_image.to(args.device)
        # gt_mask = viewpoint_cam.mask.to(args.device).unsqueeze(0)
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background       
        

        # stage 1: init
        if process_state == "init":
            gaussians_init.update_learning_rate(iteration)
            gaussians_init.reset_neighbors()
            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians_init.oneupSHdegree()

            # render

            render_pkg = render(viewpoint_cam, gaussians_init, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
            image,  depth, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            if viewpoint_cam.alpha_mask is not None:
                        alpha_mask = viewpoint_cam.alpha_mask.to(args.device)
                        image *= alpha_mask
            
            # Basic Loss
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
            
            # Depth regularization
            Ll1depth_pure = 0.0
            invDepth = None
            mono_invdepth = None
            if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
                invDepth = render_pkg["depth"]
                mono_invdepth = viewpoint_cam.invdepthmap.to(args.device)
                depth_mask = viewpoint_cam.depth_mask.to(args.device)

                Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
                Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
                loss += Ll1depth
                Ll1depth = Ll1depth.item()
            else:
                Ll1depth = 0

            # regularizer
            
            loss_align = 0
            loss_overlap = 0
            if args.reg_align or args.reg_overlap:
                gaussians_init.reset_neighbors()
                neighbor_idx  = gaussians_init.get_neighbors_of_random_points(gaussians_init.get_xyz.shape[0] // 10)
            if args.reg_align:
                # gaussian_align = gaussians_init.compute_gaussian_alignment_with_neighbors(neighbor_idx)
                # loss_align = (1-gaussian_align).mean()
                loss_align = align_loss(stprs, neighbor_idx)
                loss+= loss_align * opt.lambda_align
            if args.reg_overlap:
                gaussian_overlap = gaussians_init.compute_gaussian_overlap_with_neighbors(neighbor_idx)
                loss_overlap = gaussian_overlap.mean()
                loss += loss_overlap * opt.lambda_overlap
            loss.backward()
            
        elif process_state == "stprs" :
            stprs.update_learning_rate(iteration)
            stprs.reset_neighbors()
            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                stprs.oneupSHdegree()

            # render
            render_pkg = render(viewpoint_cam, stprs, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
            # render_pkg = render_with_depth(viewpoint_cam, stprs, pipe, bg)
            image_stprs, depth_stpr, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            image = image_stprs
            if viewpoint_cam.alpha_mask is not None:
                alpha_mask = viewpoint_cam.alpha_mask.to(args.device)
                image_stprs *= alpha_mask
            
            # Basic Loss
            Ll1 = l1_loss(image_stprs, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image_stprs.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image_stprs, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
            # Depth regularization
            Ll1depth_pure = 0.0
            invDepth = None
            mono_invdepth = None
            if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
                invDepth = render_pkg["depth"]
                mono_invdepth = viewpoint_cam.invdepthmap.to(args.device)
                depth_mask = viewpoint_cam.depth_mask.to(args.device)

                Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
                Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
                loss += Ll1depth
                Ll1depth = Ll1depth.item()
            else:
                Ll1depth = 0
            
            # mask loss
            # loss_mask = l1_loss(mask, gt_mask)
            # loss += loss_mask * opt.lambda_mask


            if args.reg_align or args.reg_overlap:
                stprs.reset_neighbors()
                neighbor_idx  = stprs.get_neighbors_of_random_points(stprs.get_xyz.shape[0] )
            if args.reg_align:
                # gaussian_align = stprs.compute_gaussian_alignment_with_neighbors(neighbor_idx)
                # loss_align = (1-gaussian_align).mean()
                loss_align = align_loss(stprs, neighbor_idx)
                loss+= loss_align * opt.lambda_align
            if args.reg_overlap:
                gaussian_overlap = stprs.compute_gaussian_overlap_with_neighbors(neighbor_idx)
                loss_overlap = (1-gaussian_overlap.mean())
                loss += loss_overlap * opt.lambda_overlap
            if args.reg_freq:
                loss_freq = stprs.low_freq_loss()
                loss += loss_freq * opt.lambda_freq
            loss.backward()
            
        elif process_state == "appgs":
            # appgs.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                appgs.oneupSHdegree()
                
            render_pkg_stprs = render(viewpoint_cam, stprs, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
            render_pkg_appgs = render(viewpoint_cam, appgs, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
            image_stprs, depth_stprs, viewspace_point_tensor_stprs, visibility_filter_stprs, radii_stprs = render_pkg_stprs["render"], render_pkg_stprs["depth"], render_pkg_stprs["viewspace_points"], render_pkg_stprs["visibility_filter"], render_pkg_stprs["radii"]
            image, depth_appgs, viewspace_point_tensor, visibility_filter, radii = render_pkg_appgs["render"], render_pkg_appgs["depth"], render_pkg_appgs["viewspace_points"], render_pkg_appgs["visibility_filter"], render_pkg_appgs["radii"]
            if viewpoint_cam.alpha_mask is not None:
                alpha_mask = viewpoint_cam.alpha_mask.to(args.device)
                image *= alpha_mask
            # Basic Loss
            # Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)
            Ll1 = l1_loss(image_stprs, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
            # Depth regularization
            Ll1depth_pure = 0.0
            invDepth = None
            mono_invdepth = None
            if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
                invDepth = render_pkg_stprs["depth"]
                mono_invdepth = viewpoint_cam.invdepthmap.to(args.device)
                depth_mask = viewpoint_cam.depth_mask.to(args.device)

                Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
                Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
                loss += Ll1depth * 0.001
                Ll1depth = Ll1depth.item()
            else:
                Ll1depth = 0
            
            # mask loss
            loss_mask = 0
            # if args.reg_mask:
                # loss_mask_app = l1_loss(mask_appgs, gt_mask)
                # loss += loss_mask_app * opt.lambda_mask
                # loss_mask_stprs = l1_loss(mask_stprs, gt_mask)
                # loss += loss_mask_stprs * opt.lambda_mask

            # regularizer           

            if args.reg_align or args.reg_overlap:
                # appgs.reset_neighbors()
                stprs.reset_neighbors()
                neighbor_idx  = stprs.get_neighbors_of_random_points(-1)
                # neighbor_idx_appg  = appgs.get_neighbors_of_random_points(stprs.get_xyz.shape[0]//10)
            if args.reg_align:
                # gaussian_align = stprs.compute_gaussian_alignment_with_neighbors(neighbor_idx)
                # loss_align = (1-gaussian_align).mean()
                loss_align = align_loss(stprs, neighbor_idx)
                loss+= loss_align * opt.lambda_align
            if args.reg_overlap:
                gaussian_overlap = stprs.compute_gaussian_overlap_with_neighbors(neighbor_idx)
                loss_overlap = gaussian_overlap.mean()
                loss += loss_overlap * opt.lambda_overlap
            if args.reg_opacity:
                loss_opacity_app = appgs.opacity_regularizer()
                loss_opacity_stprs = stprs.opacity_regularizer()
                loss += loss_opacity_app * opt.lambda_opacity + loss_opacity_stprs * opt.lambda_opacity
                
            # binding loss
            # nn = gaussians_init.get_nn_between_appgs_and_stprs() # (num_appgs, 1)
            # cylinder_paras = stprs.convert_gs_to_cylinders()
            # loss_bind = appgs.gs_cylinder_distance(nn, cylinder_paras)
            cylinder_paras = stprs.convert_gs_to_cylinders()
            appgs = gaussians_init.build_appgs_from_cylinder(cylinder_params=cylinder_paras, num_samples=100)
            points = appgs.get_xyz
            loss_bind = chamfer_distance(points.unsqueeze(0),gt_xyz.unsqueeze(0))[0]
            loss += loss_bind * 0.1
            loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                if appgs is not None:
                    progress_bar.set_postfix({"L1": f"{Ll1.item():.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}",  "Num of appgs":f"{appgs.get_xyz.shape[0]}", "Num of stprs":f"{stprs.get_xyz.shape[0]}","Align Loss": f"{loss_align:.{7}f}", 
                                              "Overlap Loss": f"{loss_overlap:.{7}f}",})
                else:
                    progress_bar.set_postfix({"L1": f"{Ll1.item():.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}", "Align Loss": f"{loss_align:.{7}f}", 
                                          "Overlap Loss": f"{loss_overlap:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if appgs is not None:
                num_appgs = appgs.get_xyz.shape[0]
            if stprs is not None:
                num_stprs = stprs.get_xyz.shape[0]
            training_report(tb_writer, iteration, Ll1, loss, Ll1depth_pure,loss_align, loss_overlap,l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp, image,  invDepth,mono_invdepth, image_stprs,loss_freq, num_appgs, num_stprs, loss_opacity_app, loss_opacity_stprs, loss_bind)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration,save_stpr=True)

            # Densification
            if iteration < opt.densify_until_iter:
                if process_state == "init":
                # Keep track of max radii in image-space for pruning
                    gaussians_init.max_radii2D[visibility_filter] = torch.max(gaussians_init.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians_init.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians_init.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                       
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians_init.reset_opacity()
                
                elif process_state == "stprs":
                    stprs.max_radii2D[visibility_filter] = torch.max(stprs.max_radii2D[visibility_filter], radii[visibility_filter])
                    stprs.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval:
                        if stprs._xyz.shape[0]<=args.max_stpr_num:
                            size_threshold = 30 if iteration > opt.opacity_reset_interval else None # size_threshold:20
                            stprs.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii, size_threshold_small=None)
                        else:
                            size_threshold = 30 if iteration > opt.opacity_reset_interval else None
                            stprs.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent*10, size_threshold, radii, only_prune=True)
                    if iteration % opt.opacity_reset_interval == 0:
                        stprs.reset_opacity_stpr()


                elif process_state == "appgs":
                    # appgs.max_radii2D[visibility_filter] = torch.max(appgs.max_radii2D[visibility_filter], radii[visibility_filter])
                    # appgs.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    stprs.max_radii2D[visibility_filter_stprs] = torch.max(stprs.max_radii2D[visibility_filter_stprs], radii_stprs[visibility_filter_stprs])
                    stprs.add_densification_stats(stprs._xyz, visibility_filter_stprs)
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        # appgs.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold,radii)
                        # appgs.drop_low_opacity_points()
                        # appgs.reset_opacity_stpr()
                        size_threshold_large, size_threshold_small = 30, 5

                        if stprs.get_xyz.shape[0] < args.max_stpr_num:
                            stprs.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii_stprs)
                            # if stprs.get_xyz.shape[0] > opt.max_stpr_num//2:
                            # stprs.drop_low_opacity_points()
                            # stprs.reset_opacity_stpr()
                            stprs.reset_neighbors()
                            neighbor_idx = stprs.get_neighbors_of_random_points(-1)
                            gaussian_overlap = stprs.compute_gaussian_overlap_with_neighbors(neighbor_idx)
                            # stprs.merge_gaussians(gaussian_overlap, neighbor_idx)
                            stprs.drop_low_opacity_points()
                        else:
                            stprs.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent*10, size_threshold, radii_stprs,only_prune=True)
                            neighbor_idx = stprs.get_neighbors_of_random_points(-1)
                            gaussian_overlap = stprs.compute_gaussian_overlap_with_neighbors(neighbor_idx)
                            # stprs.merge_gaussians(gaussian_overlap, neighbor_idx)
                            stprs.drop_low_opacity_points() 
                    # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    #     stprs.drop_low_opacity_points()
                    #     stprs.reset_opacity()

            # if iteration >  opt.iter_build_appgs and iteration%opt.iter_rebind == 0:
            #     #TODO merge stprs
            #     cylinder_paras = stprs.convert_gs_to_cylinders()
            #     appgs = gaussians_init
            #     # densify stprs
            #     stprs.max_radii2D[visibility_filter_stprs] = torch.max(stprs.max_radii2D[visibility_filter_stprs], radii_stprs[visibility_filter_stprs])
            #     stprs.add_densification_stats(stprs._xyz, visibility_filter_stprs)
            #     size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #     stprs.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii_stprs)
            #     process_state = "stprs"
               
            # Optimizer step
            if iteration < opt.iterations:
                if process_state == "init":
                    gaussians_init.optimizer.step()
                    gaussians_init.optimizer.zero_grad(set_to_none = True)
                if process_state == "stprs":
                    stprs.optimizer.step()
                    stprs.optimizer.zero_grad(set_to_none = True)
                if process_state == "appgs":
                    # appgs.optimizer.step()
                    # appgs.optimizer.zero_grad(set_to_none = True)
                    stprs.optimizer.step()
                    stprs.optimizer.zero_grad(set_to_none = True)


            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                # torch.save((appgs.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + "_appgs.pth")
                torch.save((stprs.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + "_stprs.pth")

def prepare_output_and_logger(args):   
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, loss_depth,loss_align,loss_overlap,l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp, image, depth,mono_invdepth,image_stprs,loss_freq, num_appgs, num_stprs, loss_opacity_appgs, loss_opacity_stprs, loss_bind):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/depth_loss', loss_depth, iteration)
        tb_writer.add_scalar('train_loss_patches/align_loss', loss_align, iteration)
        tb_writer.add_scalar('train_loss_patches/overlap_loss', loss_overlap, iteration)
        tb_writer.add_scalar('train_loss_patches/freq_loss', loss_freq, iteration)
        tb_writer.add_scalar('train_loss_patches/num_appgs', num_appgs, iteration)
        tb_writer.add_scalar('train_loss_patches/num_stprs', num_stprs, iteration)
        tb_writer.add_scalar('train_loss_patches/opacity_loss_appgs', loss_opacity_appgs, iteration)
        tb_writer.add_scalar('train_loss_patches/opacity_loss_stprs', loss_opacity_stprs, iteration)
        tb_writer.add_scalar('train_loss_patches/binding_loss', loss_bind, iteration)
        
        # add image 
        tb_writer.add_images('train_image_patches/render', image[None], global_step=iteration)
        if depth is not None:
            tb_writer.add_images('train_image_patches/depth', depth[None], global_step=iteration)
        if mono_invdepth is not None:
            tb_writer.add_images('train_image_patches/mono_invdepth', mono_invdepth[None], global_step=iteration)
        if image_stprs is not None:
            tb_writer.add_images('train_image_patches/stprs_render', image_stprs[None], global_step=iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
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

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000,7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3000,7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[3000,7000,15000,30000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')
    parser.add_argument("--reg_mask", action="store_true", default=False)
    parser.add_argument("--reg_align", action="store_true", default=False)
    parser.add_argument("--reg_overlap", action="store_true", default=False)
    parser.add_argument("--reg_freq", action="store_true", default=False)
    parser.add_argument("--reg_opacity", action="store_true", default=False)
    parser.add_argument("--max_stpr_num", type=int, default=50000 )

    
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
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,args)

    # All done
    print("\nTraining complete.")

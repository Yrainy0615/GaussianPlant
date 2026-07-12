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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self.root_path = "/mnt/data"
        self._model_path = ""
        self._images = "images"
        # self.pretrain_path = "feature_pretrain"
        self._depths = ""
        self.mask_path = "masks"
        self.feature_path = "dinov3_dim128"
        self.f_name = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = True
        self.speedup = False
        self.render_items = ['RGB', 'Depth', 'Edge', 'Normal', 'Curvature', 'Feature Map']

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005 # 0.005
        self.label_lr = 0.001
        self.percent_dense = 0.01
        self.semantic_feature_lr = 0.001 
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.lambda_dssim = 0.2
        self.lambda_op = 0.05
        self.lambda_size = 0.5    # penalise oversized StrPr (within the overlap term)
        self.lambda_graph = 10
        self.lambda_bind = 0.1
        self.lambda_align = 0.1
        self.lambda_axis = 1.0    # branch StrPr long-axis -> branch-tangent alignment
        self.lambda_branch_photo = 0.5  # branch-only photometric: match branch StrPr render to AppGS in branch-alpha coverage
        self.lambda_scale_hinge = 5.0   # one-sided penalty on StrPr world scale ABOVE prune_scale_frac*extent (suppress ambient-wash blobs)
        self.lambda_sem = 0.001
        self.lambda_col = 1.0     # L_col: pull StrPr label toward (fixed) color-consistent class
        self.lambda_conf = 0.05   # L_conf: p(1-p) confidence (deprecated; collapses -> not wired)
        self.lambda_bfrac = 2.0   # branch-fraction prior: keep mean(p) ~ branch_frac (anti-collapse)
        self.lambda_label = 1.0   # pull StrPr label toward the shape(dimensionality)+colour target
        self.w_shape = 2.0        # weight of the binding dimensionality cue (linearity-planarity)
        self.w_col_lab = 2.0      # weight of the colour cue (brown=branch) in the label target
        self.densification_interval = 100
        self.opacity_reset_interval = 1000
        self.densify_from_iter = 500
        self.densify_until_iter = 5000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        # self.max_stpr_num = 2000
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

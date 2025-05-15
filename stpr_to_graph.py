import os
import torch
from scene.gaussian_model import GaussianModel


if __name__ == "__main__":
    stpr_file = 'output/ficus_debug/grad_0.00075_max3000_sl_d_random_label_wo_appbind/point_cloud/iteration_7000/stprs.ply'
    stprs = GaussianModel(sh_degree=3)
    stprs.load_ply(stpr_file)
    branch_mask = [torch.sigmoid(stprs.label)>0.5 ]
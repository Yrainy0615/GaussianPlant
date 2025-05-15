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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from plyfile import PlyData, PlyElement
import numpy as np
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()



def align_loss(gs, neighbor_index):
    """
    align neighboring gaussians' orientation and scale
    """
    point_idx = neighbor_index[:,0]
    neighbor_index = neighbor_index[:,1:]

    quat_samples = gs.get_rotation[point_idx]
    quat_neighbours = gs.get_rotation[neighbor_index]
    # scale loss
    scale_samples = gs.get_scaling[point_idx]
    scale_neighbours = gs.get_scaling[neighbor_index]
    scale_diff = (scale_samples[:,None] - scale_neighbours) + 1e-4
    loss_scale = scale_diff.norm(dim=-1).mean()
    
    # orientation loss #TODO: fix bug
    # try add norm 
    quat_samples = torch.nn.functional.normalize(quat_samples,dim=-1)
    quat_neighbours = torch.nn.functional.normalize(quat_neighbours,dim=-1)
    quat_dot = torch.sum(quat_samples.unsqueeze(1) * quat_neighbours, dim=-1).abs()    
    loss_ori = 1 - quat_dot.mean()
    return loss_scale+loss_ori


def mst_loss(top,bottom,stpr_roataions,mst_edges):
    """
    stpr_xyz: [N,3]
    stpr_roataions: [N,4]
    mst_edges: [E,2]
    loss_gap: penalize the gap between stprs
    """
    N = top.shape[0]
    points = torch.zeros((2*N,3), device=top.device)
    points[0::2] = top
    points[1::2] = bottom    
    loss_mst = 0
    mst_edges = torch.tensor(mst_edges).to(top.device)
    
    ext_mask   = (mst_edges//2)[:,0] != (mst_edges//2)[:,1]
    edge_idx   = mst_edges[ext_mask] 
    gap = points[edge_idx[:,0]] - points[edge_idx[:,1]]
    # save paired points with the same color for test
    loss_gap = gap.mean()
    loss_mst += loss_gap
    return loss_mst


def save_paired_points(top, bottom, pairs,  # pairs = tensor([[i,j],...])
                       ply_path='paired_pts.ply'):
    """
    top, bottom : (N,3)  torch.Tensor
    pairs       : (K,2)  long  ->  (top_i , bottom_j)
    """
    K = pairs.size(0)
    device = top.device
    # ---------- 顶点坐标 ----------
    v_top  = top[pairs[:,0]]
    v_bot  = bottom[pairs[:,1]]
    verts  = torch.cat([v_top, v_bot], 0).detach().cpu().numpy().astype('f4')  # (2K,3)

    # ---------- 颜色（同对同色，随机） ----------
    rgb = (np.random.rand(K,3)*255).astype('u1')
    rgb = np.repeat(rgb, 2, axis=0)                                   # (2K,3)

    # ---------- 1-D structured vertex array ----------
    vertex = np.empty(2*K, dtype=[('x','f4'),('y','f4'),('z','f4'),
                                  ('red','u1'),('green','u1'),('blue','u1')])
    vertex['x'], vertex['y'], vertex['z'] = verts.T
    vertex['red'], vertex['green'], vertex['blue'] = rgb.T
    el_v = PlyElement.describe(vertex, 'vertex')

    # ---------- edge element ----------
    # 每对端点在 verts 中的索引： (2*i   , 2*i+1)
    edge_idx = np.arange(2*K, dtype=np.uint32).reshape(-1,2)
    edge = np.empty(K, dtype=[('vertex1','u4'),('vertex2','u4')])
    edge['vertex1'] = edge_idx[:,0]
    edge['vertex2'] = edge_idx[:,1]
    el_e = PlyElement.describe(edge, 'edge')

    # ---------- 写 PLY ----------
    PlyData([el_v, el_e], text=True).write(ply_path)
    print(f'Saved {ply_path}  (verts {2*K}, edges {K})')

def label_loss(label):
    p = torch.sigmoid(label)
    loss = p * (1 - p)
    return loss

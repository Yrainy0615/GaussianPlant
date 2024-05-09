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
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from utils.general_utils import reconstruct_covariance
from scipy.spatial import cKDTree

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


def local_normal_consistency_loss(gaussians, k=10):
    points = gaussians.get_xyz
    covariances = gaussians.get_covariance()
    # convert covariance matrix to 3*3
    cov3d = reconstruct_covariance(covariances)
    # calculate eigenvalues and eigenvectors of covariance matrices
    eigenvalue, eigenvector = torch.linalg.eigh(cov3d)
    pricipals = eigenvector[:, :, 2]
    normals = eigenvector[:, :, 0]

    
    def find_nearest_neighbors(points, data_points, n_neighbors):
        data_points = data_points.detach().cpu().numpy()
        points = points.detach().cpu().numpy()
        tree = cKDTree(data_points)
        distances, indices = tree.query(points, n_neighbors)
        return distances, indices
    def find_nearest_neighbors_torch(points, data_points, n_neighbors):
        dist_matrix = torch.cdist(points, data_points)
        # batch 
        batche_size = 1000
        indices = torch.zeros((dist_matrix.shape[0], n_neighbors), dtype=torch.long, device=dist_matrix.device)
        for i in range(0, dist_matrix.shape[0], batche_size):
            end = min(i + batche_size, dist_matrix.shape[0])
            batch_distance = dist_matrix[i:end]
            distances_batch, indices_batch = torch.topk(batch_distance, n_neighbors, largest=False)
            indices[i:end] = indices_batch
        return indices
    indices = find_nearest_neighbors_torch(points, points, n_neighbors=k)

    # normal consistency loss
    query_normals = normals[indices[:, 0]]
    neighbor_normals = normals[indices[:, 1:].reshape(-1)].view(indices.shape[0], indices.shape[1] - 1, 3)
    dot_products = (query_normals.unsqueeze(1) * neighbor_normals).sum(dim=2)
    norm_query_normals = query_normals.norm(p=2, dim=1, keepdim=True)
    norm_neighbor_normals = neighbor_normals.norm(p=2, dim=2)
    cos_angles = dot_products / (norm_query_normals * norm_neighbor_normals + 1e-5)
    normal_consistency_scores = cos_angles.abs().mean(dim=1)
    loss = l2_loss(normal_consistency_scores, torch.ones_like(normal_consistency_scores)) 
    return loss
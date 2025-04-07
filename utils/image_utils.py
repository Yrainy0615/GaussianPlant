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
import cv2
import numpy as np
import os
import torch
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize



def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def save_tensor_as_image(tensor, save_name):
    # Move tensor to CPU, detach from computation graph
    tensor = tensor.detach().cpu().float()
    if tensor.dim()==2:
        tensor = tensor.unsqueeze(0)
    
        
    # Normalize values to [0,1] if not already in range
    if tensor.min() < 0 or tensor.max() > 1:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

    # Convert 1-channel images to 3-channel for better visualization
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)  # Convert grayscale to RGB

    # Convert tensor to numpy array (H, W, C) format for saving
    image_numpy = tensor.permute(1, 2, 0).numpy()

    # Save image
    plt.imsave(save_name, image_numpy)

    print(f"Saved tensor as image: {save_name}")
    
    

def build_laplacian_pyramid(image, num_levels=4):
    # 构建高斯金字塔
    gaussian_pyramid = [image.copy()]
    for i in range(num_levels):
        image = cv2.pyrDown(image)
        gaussian_pyramid.append(image)

    # 构建拉普拉斯金字塔
    laplacian_pyramid = []
    for i in range(num_levels, 0, -1):
        GE = cv2.pyrUp(gaussian_pyramid[i])
        GE = cv2.resize(GE, (gaussian_pyramid[i-1].shape[1], gaussian_pyramid[i-1].shape[0]))  # 尺寸对齐
        L = cv2.subtract(gaussian_pyramid[i-1], GE)
        laplacian_pyramid.append(L)

    return laplacian_pyramid

def show_laplacian_pyramid(lap_pyr):
    plt.figure(figsize=(15, 5))
    for i, layer in enumerate(lap_pyr):
        layer_show = cv2.normalize(layer, None, 0, 255, cv2.NORM_MINMAX)
        plt.subplot(1, len(lap_pyr), i+1)
        # plt.imshow(cv2.cvtColor(layer_show.astype(np.uint8), cv2.COLOR_BGR2RGB))
        plt.title(f"Laplacian Level {i}")
        plt.axis('off')
        plt.imsave(f"laplacian_level_{i}_blur.png", layer_show)

    plt.tight_layout()

if __name__ == "__main__":
    # Example usage
    # image = cv2.imread("data/plant3/images/0009.png")
    image = cv2.imread("blur.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    lap_pyr = build_laplacian_pyramid(image, num_levels=4)
    show_laplacian_pyramid(lap_pyr)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(image, (31, 31), sigmaX=10, sigmaY=10)
    # cv2.imwrite('blur.png', blur)

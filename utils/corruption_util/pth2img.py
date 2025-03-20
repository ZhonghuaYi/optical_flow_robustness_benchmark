import torch
import scipy.io as scio
import cv2
import numpy as np
from torchvision import utils
import torch.nn as nn
import math
import torch.nn.functional as F
from torchvision.transforms.functional import rotate, InterpolationMode, resize
import time
import os
from torchvision import io

"""
输入：
    psf:[N×C×H1×W1]的tensor
    img:[N2×C×H2×W2]的tensor
    h_num：纵向图像块数量
    w_num：横向图像块数量
    patch_length：图像块尺寸大小
    device：设备
    view_pos：[N×1]的tensor，表示psf数组中每个视场的位置，比如[0,0.2,0.4,0.8,1.0]
    fast: 默认false，为True代表完全并行，为False代表部分并行
输出：
    blur_img：[N2×C×H2×W2]的tensor
"""


def super_fast_patch_wise_conv(psf, img, img_size=(384, 1248), h_num=100, w_num=100, patch_length=64
                               , device=None, views_pos=None, fast=False):
    time1 = time.time()
    N, C, H, W = img.shape
    pad_whole = (psf.size(3) - 1) // 2  # 大图像边缘填充尺寸
    pad = (psf.size(3) - 1) // 2  # 图像块边缘填充尺寸

    # 大图像边缘填充
    img_pad = F.pad(img, (pad_whole, pad_whole, pad_whole, pad_whole), mode='reflect').to(device)

    # 图像分块，分块尺寸为patch_length + 2 * pad，这样卷积后尺寸刚好为patch_length
    inputs_pad = F.unfold(img_pad, patch_length + 2 * pad, 1, 0, patch_length).transpose(1, 2).reshape(N,
                                                                                                       C * h_num * w_num,
                                                                                                       patch_length + 2 * pad,
                                                                                                       patch_length + 2 * pad)
    # 批量计算每个图像块的视场位置以及对应的PSF
    index_h, index_w = torch.meshgrid(
        torch.linspace(0, h_num - 1, h_num, device=device),
        torch.linspace(0, w_num - 1, w_num, device=device),
        indexing='ij',
    )
    d_max = np.sqrt(h_num ** 2 + w_num ** 2) / 2
    d_loc = ((index_h + 0.5 - h_num / 2) ** 2 + (index_w + 0.5 - w_num / 2) ** 2).sqrt()
    standard_vector = torch.Tensor([0, 1]).to(device)
    yloc = index_h + 0.5 - h_num / 2
    xloc = index_w + 0.5 - w_num / 2
    location_vector = torch.stack((xloc, yloc), dim=2)
    norm_l = torch.stack((xloc, yloc), dim=2).square().sum(dim=2).sqrt()
    cos_angle = (location_vector * standard_vector).sum(dim=2) / norm_l
    rotate_theta = torch.arccos(cos_angle) / math.pi * 180
    rotate_theta[(xloc == 0) * (yloc == 0)] = 0
    rotate_theta[xloc > 0] = -rotate_theta[xloc > 0]
    rotate_theta = rotate_theta.reshape(h_num * w_num)
    # index为views_pos全采样视场位置分别与每个图像块视场位置之差
    index = (torch.from_numpy(views_pos).unsqueeze(0).unsqueeze(0)).float().to(device) - (d_loc / d_max).unsqueeze(2)
    # 这里表示点扩散函数插值方程，也可以改成其它的，此处插值规律为呈现-4次方，但采样视场数量足够大的话这就不重要了
    index = index ** (-4)
    if fast:
        psf_used = ((index.unsqueeze(3).unsqueeze(3).unsqueeze(3) * psf.unsqueeze(0).unsqueeze(0)).sum(dim=2).squeeze(
            2)).reshape(h_num * w_num, psf.size(1), psf.size(2), psf.size(3))
    else:
        psf_used = torch.zeros(h_num * w_num, psf.size(1), psf.size(2), psf.size(3)).to(device)
        for index_h in range(0, h_num):
            for index_w in range(0, w_num):
                psf_used[index_w + index_h * w_num] = (psf.transpose(0, 3) * index[index_h, index_w]).transpose(0, 3).sum(dim=0)
    # 点扩散函数批量旋转
    psf_used1 = my_rotate(psf_used, angle=rotate_theta)
    psf_used2 = psf_used1 / psf_used1.sum(dim=3).sum(dim=2).unsqueeze(2).unsqueeze(2)
    # PSF_draw为所有视场块对应的PSF组合
    PSF_draw = psf_used2.reshape(h_num * w_num * psf.size(1), 1, psf.size(2), psf.size(3))
    # 批量卷积
    outputs = F.conv2d(inputs_pad, PSF_draw, stride=1, groups=C * h_num * w_num)
    # 卷积图像块重组为整个图像
    blur_img = F.fold(outputs.reshape(N, h_num * w_num, psf.size(1) * patch_length * patch_length).transpose(1, 2),
                      img_size, patch_length, 1, 0, patch_length)
    time2 = time.time()

    print('patch: {item}'.format(item=patch_length))
    print('super_fast_conv_time: {time}'.format(time=time2 - time1))

    return blur_img


# 自定义批量旋转函数
def my_rotate(img, angle=None):
    shear = [0, 0]
    rot = angle / 180 * math.pi
    sx, sy = [math.radians(s) for s in shear]
    # RSS without scaling
    a = torch.cos(rot - sy) / math.cos(sy)
    b = -torch.cos(rot - sy) * math.tan(sx) / math.cos(sy) - torch.sin(rot)
    c = torch.sin(rot - sy) / math.cos(sy)
    d = -torch.sin(rot - sy) * math.tan(sx) / math.cos(sy) + torch.cos(rot)
    zeros = torch.zeros_like(a)
    matrix = torch.stack((d, -b, zeros, -c, a, zeros), dim=1)
    n, c, w, h = img.shape[0], img.shape[1], img.shape[3], img.shape[2]
    ow, oh = (w, h)
    theta = matrix.reshape(-1, 2, 3)
    d = 0.5
    base_grid = torch.empty(n, oh, ow, 3, dtype=theta.dtype, device=theta.device)
    x_grid = torch.linspace(-ow * 0.5 + d, ow * 0.5 + d - 1, steps=ow, device=theta.device)
    base_grid[..., 0].copy_(x_grid)
    y_grid = torch.linspace(-oh * 0.5 + d, oh * 0.5 + d - 1, steps=oh, device=theta.device).unsqueeze_(-1)
    base_grid[..., 1].copy_(y_grid)
    base_grid[..., 2].fill_(1)
    rescaled_theta = theta.transpose(1, 2) / torch.tensor([0.5 * w, 0.5 * h], dtype=theta.dtype, device=theta.device)
    output_grid = base_grid.view(n, oh * ow, 3).bmm(rescaled_theta)
    grid = output_grid.view(n, oh, ow, 2)
    output = torch.grid_sampler(img, grid, 0, 0, False)
    return output


if __name__ == '__main__':
    patch_size = 16
    img_size = (384, 1248)
    dev = 0
    device = torch.device('cuda:' + str(dev))
    h_num = img_size[0] // patch_size
    w_num = img_size[1] // patch_size
    input_psf_dir = r'./yizhonghua/psf/'
    input_img_dir = r'./yizhonghua/input/'
    output_img_dir = r'./yizhonghua/output/'
    img_s = os.listdir(input_img_dir)
    psf_s = os.listdir(input_psf_dir)
    for index in range(len(img_s)):
        img_path = img_s[index]
        img = (io.read_image(input_img_dir + img_path).float() / 255).unsqueeze(0).to(device)
        N, C, H, W = img.shape
        pad_h1 = (img_size[0] - H) // 2
        pad_h2 = img_size[0] - H - pad_h1
        pad_w1 = (img_size[1] - W) // 2
        pad_w2 = img_size[1] - W - pad_w1
        img_pad = F.pad(img, (pad_w1, pad_w2, pad_h1, pad_h2), mode='reflect').to(device)
        for j in range(len(psf_s)):
            psf_path = psf_s[j]
            psf = torch.load(input_psf_dir + psf_path, map_location='cuda:0')
            fov_pos = torch.linspace(0, 1, len(psf)).detach().cpu().numpy()
            blur_img = super_fast_patch_wise_conv(psf, img_pad, img_size=img_size, h_num=h_num, w_num=w_num, patch_length=patch_size, device=device, views_pos=fov_pos, fast=True)
            blur_img = blur_img[:, :, pad_h1:-pad_h2, pad_w1: -pad_w2]
            name = psf_path[:-4] + '_' + img_path
            utils.save_image(blur_img, output_img_dir + name)


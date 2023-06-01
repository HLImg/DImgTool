# -*- coding: utf-8 -*-
# @Time    : 5/31/23 11:04 AM
# @File    : image_loss.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn


import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
from source.loss.basic_loss.pixel_loss import CharbonnierLoss
from torchvision.models import vgg16, vgg16_bn, vgg19, vgg19_bn

warnings.filterwarnings("ignore")


class PerceptualLoss(nn.Module):
    """
    reference : https://github.com/Xinzhe99/Perceptual-Loss-for-pytorch/
    """

    def __init__(self, basic_loss, net_type, weight=1., net_indexs=None):
        super(PerceptualLoss, self).__init__()
        self.weight = weight
        self.net_indexs = net_indexs
        self._get_network_(net_type)
        self._get_criterion_(basic_loss)

    def forward(self, input, target):
        loss = 0.
        device = input.device
        for index in self.net_indexs:
            feature_module = self._get_feature_module_(index, device)
            loss += self._calculte_loss_(feature_module, input, target)
        return self.weight * loss

    def _calculte_loss_(self, feature_module, input, target):
        # print(input.device, feature_module.device)
        feature_in = feature_module(input)
        feature_tar = feature_module(target)
        loss = self.criterion(feature_tar, feature_in)
        return loss

    def _get_feature_module_(self, index, device):
        self.net.eval()
        # Freezing parameters
        for param in self.net.parameters():
            param.requires_grad = False
        feature_module = self.net[0: index + 1]
        feature_module = feature_module.to(device)
        return feature_module

    def _get_network_(self, net_type):
        if net_type == "vgg16":
            self.net = vgg16(pretrained=True, progress=True).features
        elif net_type == "vgg16_bn":
            self.net = vgg16_bn(pretrained=True, progress=True).features
        elif net_type == "vgg19":
            self.net = vgg19(pretrained=True, progress=True).features
        elif net_type == "vgg19_bn":
            self.net = vgg19_bn(pretrained=True, progress=True).features
        else:
            assert 1 == 2, f"pretrained perceptual network named {net_type} is not exists."

    def _get_criterion_(self, loss_name):
        if loss_name.lower() == "l1":
            self.criterion = F.l1_loss
        elif loss_name.lower() == "mse":
            self.criterion = F.mse_loss
        elif loss_name.lower() == "char":
            self.criterion = CharbonnierLoss(weight=1, eps=1e-3, reduction="mean")
        else:
            self.criterion = F.l1_loss


class EdgeLoss(nn.Module):
    """
    reference : https://github.com/swz30/MPRNet
    """

    def __init__(self, basic_loss, weight=1.):
        super(EdgeLoss, self).__init__()
        self.weight = weight
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel_gray = torch.matmul(k.t(), k).unsqueeze(0).repeat(1, 1, 1, 1)
        self.kernel_rgb = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        self._get_criterion_(basic_loss)

    def forward(self, input, target):
        self.kernel = self.kernel_gray.to(input.device) if input.shape[1] == 1 else self.kernel_rgb.to(input.device)
        laplacian_in = self.laplacian_kernel(input)
        laplacian_tar = self.laplacian_kernel(target)
        loss = self.criterion(laplacian_tar, laplacian_in)
        return self.weight * loss

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(input=img, pad=(kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(input=img, weight=self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        # downsample
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        filtered = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def _get_criterion_(self, loss_name):
        if loss_name.lower() == "l1":
            self.criterion = F.l1_loss
        elif loss_name.lower() == "mse":
            self.criterion = F.mse_loss
        elif loss_name.lower() == "char":
            self.criterion = CharbonnierLoss(weight=1, eps=1e-3, reduction="mean")
        else:
            self.criterion = F.l1_loss
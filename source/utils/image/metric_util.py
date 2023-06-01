# -*- coding: utf-8 -*-
# @Time    : 5/30/23 6:59 PM
# @File    : metric_util.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import lpips
import cv2 as cv
import numpy as np
import torch
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio


class TensorMetric:
    def __init__(self, border, net_type=None, color="ycbcr"):
        self.mode = color
        self.border = border
        if not net_type:
            self.lpips = lambda x, y: 0
        else:
            self.lpips = lpips.LPIPS(pretrained=True, net=net_type)

    def tensor2image(self, tensor):
        """
        tensor to image
        @param tensor: (c, h, w)
        @return: (h, w, c)
        """
        image = tensor.data.cpu().numpy().clip(0, 1)
        image = np.transpose(image, (1, 2, 0))
        h, w = image.shape[:2]
        image = image[self.border: h - self.border, self.border: w - self.border]
        if self.mode == "ycbcr":
            image = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)[:, :, :1]
        return image

    def calculate_psnr(self, input, target):
        """
        calculate psnr for tensor
        @param input: tensor(b, c, h, w)
        @param target: tensor(b, c, h, w)
        @return: sum( psnr )
        """
        assert input.shape == target.shape, "error in tensor shape"
        psnr = 0.0
        for i in range(input.shape[0]):
            psnr += peak_signal_noise_ratio(self.tensor2image(input[i]),
                                            self.tensor2image(target[i]))
        return psnr

    def calculate_ssim(self, input, target):
        """
        calculate ssim for tensor
        @param input: tensor(b, c, h, w)
        @param target: tensor(b, c, h, w)
        @return: sum( ssim )
        """
        assert input.shape == target.shape, "error in tensor shape"
        ssim = 0.0
        for i in range(input.shape[0]):
            ssim += structural_similarity(self.tensor2image(input[i]),
                                          self.tensor2image(target[i]), channel_axis=2, data_range=1)
        return ssim

    def calculate_lpips(self, input, target):
        assert input.shape == target.shape, "error in tensor shape"
        assert input.device == target.device, "error in tensor device"
        device = input.device
        self.lpips = self.lpips.to(device)
        lpips = self.lpips(input, target)
        return torch.sum(lpips).item()


def calculate_psnr(input, target, border=0, channel="ycbcr"):
    """
    calculate psnr for tensor
    @param input: tensor(b, c, h, w)
    @param target: tensor(b, c, h, w)
    @param border: int {0, 1, 2, 3}
    @param channel: "ycbcr" or "rgb"
    @return: sum(psnr)
    """
    assert input.shape == target.shape, "error in tensor shape"
    psnr = 0.0
    for i in range(input.shape[0]):
        recon, gt = input[i], target[i]
        gt = gt.data.cpu().numpy().clip(0, 1)
        recon = recon.data.cpu().numpy().clip(0, 1)
        gt = np.transpose(gt, (1, 2, 0))
        recon = np.transpose(recon, (1, 2, 0))
        h, w = recon.shape[:2]
        gt = gt[border: h - border, border: w - border]
        recon = recon[border: h - border, border: w - border]

        if channel.lower() == 'ycbcr':
            gt = cv.cvtColor(gt, cv.COLOR_RGB2YCrCb)[:, :, :1]
            recon = cv.cvtColor(recon, cv.COLOR_RGB2YCrCb)[:, :, :1]

        psnr += peak_signal_noise_ratio(gt, recon)
    return psnr


def calculate_ssim(input, target, border=0, channel="ycbcr"):
    """
    calculate psnr for tensor
    @param input: tensor(b, c, h, w)
    @param target: tensor(b, c, h, w)
    @param border: int {0, 1, 2, 3}
    @param channel: "ycbcr" or "rgb"
    @return: sum( ssim )
    """
    assert input.shape == target.shape, "error in tensor shape"
    ssim = 0.0
    for i in range(input.shape[0]):
        recon, gt = input[i], target[i]
        gt = gt.data.cpu().numpy().clip(0, 1)
        recon = recon.data.cpu().numpy().clip(0, 1)
        gt = np.transpose(gt, (1, 2, 0))
        recon = np.transpose(recon, (1, 2, 0))
        h, w = recon.shape[:2]
        gt = gt[border: h - border, border: w - border]
        recon = recon[border: h - border, border: w - border]

        if channel.lower() == 'ycbcr':
            gt = cv.cvtColor(gt, cv.COLOR_RGB2YCrCb)[:, :, :1]
            recon = cv.cvtColor(recon, cv.COLOR_RGB2YCrCb)[:, :, :1]

        ssim += structural_similarity(gt, recon, channel_axis=2, data_range=1)
    return ssim

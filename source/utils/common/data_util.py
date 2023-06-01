# -*- coding: utf-8 -*-
# @Time    : 5/30/23 6:48 PM
# @File    : data_util.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import os
import glob
import sys

import torch
import random
import cv2 as cv
import numpy as np
from cv2 import rotate
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from source.utils.image.metric_util import TensorMetric


def random_augment(*images, flip=True, rot=True):
    """
    random augment images, including flip{lr, up} and rot90x{0, 1, 2, 3}
    @param images: im1, im2, im3, ...
    @param flip: False or True
    @param rot: False or True
    @return: [im1, im2, im3, ...]
    """
    rot_k = random.randint(0, 3)
    flip_mode = random.randint(0, 3)
    augmented = []

    def _flip_(img, mode=0):
        if mode == 0:
            return img
        if mode == 1:
            return np.flip(img)
        if mode == 2:
            return np.fliplr(img)
        if mode == 3:
            return np.flipud(img)

    def _rot_(img, k=0):
        return np.rot90(img, k)

    for img in images:
        if flip:
            img = _flip_(img, flip_mode)
        if rot:
            img = _rot_(img, rot_k)
        augmented.append(img)
    return augmented


def get_image_path(directory):
    assert os.path.exists(directory), f"{directory} is not exits"
    paths = [path for path in glob.glob(os.path.join(directory, "*"))]
    paths = np.array(sorted(paths))
    return paths


def image2tensor(image):
    assert image is not None, "the image is none"
    if len(image.shape) == 2:
        image = image[np.newaxis, :]
    else:
        image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image.copy())
    return image


def tensor2image(tensor):
    assert len(tensor.shape) == 3 or len(tensor.shape) == 4, f"tensor shape is {tensor.shape}"
    image = tensor.data.cpu().numpy()
    if len(image.shape) == 4:
        image = np.transpose(image, (0, 2, 3, 1))
    else:
        image = np.transpose(image, (1, 2, 0))
    return image


def random_image2patch(*images, patch_size):
    """
    random crop images to patches,
    all images have the same resolution.
    @param images: im1, im2, im3, ....
    @param patch_size: {48, 64, 96, 128, 256}
    @return: patch_1, patch_2, patch_3, ...
    """
    h, w = images[0].shape[:2]
    ind_h = random.randint(0, max(h, patch_size) - patch_size)
    ind_w = random.randint(0, max(w, patch_size) - patch_size)
    patches = []
    for img in images:
        patch = img[ind_h: ind_h + patch_size, ind_w: ind_w + patch_size]
        patches.append(patch)
    return patches


def random_image2patch_sr(img_lr, img_hr, patch_size, up_scale=1):
    """
    random crop **two** images to patches,
    @param img_lr: low-resolution image
    @param img_hr: high-resolution image
    @param patch_size: {48, 64, 96, 128}
    @param up_scale: {1, 2, 4}
    @return: patch_lr, patch_hr
    """
    h_1, w_1 = img_lr.shape[:2]
    h_2, w_2 = img_hr.shape[:2]
    ind_h = random.randint(0, max(min(h_1, h_2 // up_scale), patch_size) - patch_size)
    ind_w = random.randint(0, max(min(w_1, w_2 // up_scale), patch_size) - patch_size)
    patch_lr = img_lr[ind_h: ind_h + patch_size, ind_w: ind_w + patch_size]
    patch_hr = img_hr[ind_h * up_scale: (ind_h + patch_size) * up_scale, ind_w * up_scale: (ind_w + patch_size) * up_scale]
    return patch_lr, patch_hr


def save_tensor_single(tensor, save_path):
    image = tensor2image(tensor)
    image = image.squeeze()
    assert len(image.shape) <= 3, "there are multi image, it can not be saved as single image."
    image = img_as_ubyte(image.clip(0, 1))
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    cv.imwrite(save_path, image)

def save_tensor_contrast(input, target, output, border, channel, save_path, dpi=400):
    tensorMetric = TensorMetric(border=border, net_type=None, color=channel)
    psnr = tensorMetric.calculate_psnr(output, target)
    ssim = tensorMetric.calculate_ssim(output, target)
    input = tensor2image(input)
    output = tensor2image(output)
    target = tensor2image(target)
    input = input.squeeze()
    output = output.squeeze()
    target = target.squeeze()
    input = img_as_ubyte(input.clip(0, 1))
    output = img_as_ubyte(output.clip(0, 1))
    target = img_as_ubyte(target.clip(0, 1))
    dif = cv.absdiff(target, output)
    h, w = output.shape[:2]
    pos_x = w // 2 - 4 * w
    pos_y = h + h // 6
    plt.figure()
    plt.subplot(141), plt.imshow(input, 'gray'), plt.title('input'), plt.axis('off')
    plt.subplot(142), plt.imshow(target, 'gray'), plt.title('target'), plt.axis('off')
    plt.subplot(143), plt.imshow(output, 'gray'), plt.title('output'), plt.axis('off')
    plt.subplot(144), plt.imshow(dif / dif.max()), plt.title('difference'), plt.axis('off')
    plt.text(pos_x, pos_y, s=f"psnr = {psnr :.6f}, ssim = {ssim : .6f}")
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def get_noise_level(levels):
    if isinstance(levels, list):
        noise_levels = levels
    elif isinstance(levels, tuple) and len(levels) == 3:
        noise_levels = np.arange(levels[0], levels[1] + levels[2], levels[2])
    else:
        noise_levels = None
        sys.exit(1)
    return noise_levels
# -*- coding: utf-8 -*-
# @Time    : 5/30/23 10:28 PM
# @File    : image_util.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import cv2 as cv
import numpy as np
from source.utils.image.blur_kernel import motion_blur_p4ip, motion_blur_deblurgan

# ------------------------------------------------------------------ #
#  Basic Image Degradation Methods : Noise, Down Sampling, Blurring  #
# ------------------------------------------------------------------ #

def add_gaussian_noise(image, level, clip=False):
    noise = np.random.normal(0, level / 255., image.shape)
    noisy = np.float32(image + noise)
    if clip:
        noisy = noisy.clip(0, 1)
    return noisy


def add_poisson_noise(image, level, clip=False):
    noisy = np.float32(np.random.poisson(image * level) / level)
    if clip:
        noisy = noisy.clip(0, 1)
    return noisy


def blur_image(image, kernel, mode='gaussian', channel='gray'):
    assert len(mode) != 0 and len(channel), 'mode is empty'
    if mode.lower() == 'opencv' or mode.lower() == 'gaussian':
        image = cv.filter2D(image, ddepth=-1, kernel=kernel)
        image = np.abs(image)
    elif mode.lower() == 'p4ip':
        image = motion_blur_p4ip(image, kernel, channel=channel)
    elif mode.lower() == 'deblurgan':
        image = motion_blur_deblurgan(image, kernel, channel)
    return image
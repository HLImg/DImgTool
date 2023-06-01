# -*- coding: utf-8 -*-
# @Time    : 5/30/23 10:27 PM
# @File    : degradation.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import source.utils.image.image_util as image_util


class AddNoisePipeLine:
    def __init__(self, type):
        self.type = type

    def __call__(self):
        if self.type.lower() == 'poisson':
            return image_util.add_poisson_noise
        elif self.type.lower() == 'gaussian':
            return image_util.add_gaussian_noise

# TODO : Image Degradation Pipeline
# including : noise (read-out, shot, compression), down-sampling, blurry
# reference SCUNet and BSRGAN
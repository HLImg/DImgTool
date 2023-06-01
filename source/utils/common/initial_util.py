# -*- coding: utf-8 -*-
# @Time    : 5/31/23 2:53 PM
# @File    : initial_util.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import torch.nn as nn
import torch.nn.init as init
from timm.models.layers import trunc_normal_


class WeightInit:
    def __init__(self, name):
        self.name = name

    def __call__(self):
        if self.name.upper() == 'swinconv'.upper():
            return self.__swinconv__
        elif self.name.upper() == 'kaiming'.upper():
            return self.__kaiming__
        elif self.name.upper() == 'swinunet'.upper():
            return self.__swinunet__

    def __swinconv__(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def __kaiming__(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if not m.bias is None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def __swinunet__(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            m.weight.data *= 1  # for residual block
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d) or isinstance(
                m, nn.ConvTranspose3d):
            init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            m.weight.data *= 1  # for residual block
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(
                m, nn.BatchNorm3d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)
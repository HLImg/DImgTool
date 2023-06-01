# -*- coding: utf-8 -*-
# @Time    : 5/30/23 11:05 PM
# @File    : pair_load.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import numpy as np
import source.utils.common.data_util as data_util
from source.dataset.basic_dataset.pair_dataset import BasicDataSetPair


class TrainDataSet(BasicDataSetPair):
    def __init__(self, input_dir, target_dir, patch_size, up_scale=1, color='gray', aug_rot=True, aug_flip=True):
        super(TrainDataSet, self).__init__(input_dir, target_dir, color=color, aug_flip=aug_flip, aug_rot=aug_rot)
        self.up_scale = up_scale
        self.patch_size = patch_size

    def __getitem__(self, item):
        input, target = self.load_image(item)
        input, target = data_util.random_augment(input, target, flip=self.aug_flip, rot=self.aug_rot)
        patch_in, patch_tar = data_util.random_image2patch_sr(input, target,
                                                              patch_size=self.patch_size, up_scale=self.up_scale)
        tensor_in = data_util.image2tensor(patch_in)
        tensor_tar = data_util.image2tensor(patch_tar)
        return tensor_in, tensor_tar


class TestDataSet(BasicDataSetPair):
    def __init__(self, input_dir, target_dir, color='gray'):
        super(TestDataSet, self).__init__(input_dir, target_dir, color)

    def __getitem__(self, item):
        input, target = self.load_image(item)
        if self.color == 'rgb' and len(input.shape) == 2:
            input = input[:, :, np.newaxis]
            target = target[:, :, np.newaxis]
            input = np.concatenate([input, input, input], axis=2)
            target = np.concatenate([target, target, target], axis=2)
        tensor_in = data_util.image2tensor(input)
        tensor_tar = data_util.image2tensor(target)
        return tensor_in, tensor_tar
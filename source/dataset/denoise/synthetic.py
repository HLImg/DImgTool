# -*- coding: utf-8 -*-
# @Time    : 5/30/23 11:07 PM
# @File    : synthetic.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import numpy as np
import source.utils.common.data_util as data_util
from source.utils.image.degradation import AddNoisePipeLine
from source.dataset.basic_dataset.single_dataset import BasicDataSetSingle


class TrainDataSet(BasicDataSetSingle):
    def __init__(self, target_dir, patch_size, levels, noise_type, color='gray',
                 clip=False, aug_flip=True, aug_rot=True):
        super(TrainDataSet, self).__init__(target_dir, color=color, aug_rot=aug_rot, aug_flip=aug_flip)
        self.clip = clip
        self.noise_levels = data_util.get_noise_level(levels)
        self.patch_size = patch_size
        self.add_noise = AddNoisePipeLine(noise_type)()

    def __getitem__(self, item):
        level = np.random.choice(self.noise_levels)
        target = self.load_image(item)
        target, = data_util.random_augment(target, flip=self.aug_flip, rot=self.aug_rot)
        patch_tar,  = data_util.random_image2patch(target, patch_size=self.patch_size)
        patch_in = self.add_noise(patch_tar.copy(), level, clip=self.clip)
        tensor_in = data_util.image2tensor(patch_in)
        tensor_tar = data_util.image2tensor(patch_tar)
        return tensor_in, tensor_tar


class TestDataSet(BasicDataSetSingle):
    def __init__(self, target_dir, levels, noise_type, clip, color='gray'):
        super(TestDataSet, self).__init__(target_dir, color)
        self.clip = clip
        self.noise_levels = data_util.get_noise_level(levels)
        self.add_noise = AddNoisePipeLine(noise_type)()

    def __getitem__(self, item):
        level = np.random.choice(self.noise_levels)
        target = self.load_image(item)
        input = self.add_noise(target.copy(), level, self.clip)
        tensor_in = data_util.image2tensor(input)
        tensor_tar = data_util.image2tensor(target)
        return tensor_in, tensor_tar
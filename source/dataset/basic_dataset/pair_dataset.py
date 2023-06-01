# -*- coding: utf-8 -*-
# @Time    : 5/30/23 10:34 PM
# @File    : pair_dataset.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import os
import h5py
import cv2 as cv
import numpy as np
from PIL import Image
from scipy.io import loadmat
from skimage import img_as_float32
from torch.utils.data import Dataset
import source.utils.common.data_util as data_util


class BasicDataSetPair(Dataset):
    def __init__(self, input_dir, target_dir, color="gray", aug_rot=True, aug_flip=True):
        super(BasicDataSetPair, self).__init__()
        self.color = color
        self.aug_rot = aug_rot
        self.aug_flip = aug_flip
        if os.path.isdir(input_dir):
            self.flag = 'opencv'
            self.input = data_util.get_image_path(input_dir)
            self.target = data_util.get_image_path(target_dir)
            self.length = len(self.input)
        else:
            if "mat" in input_dir:
                self.flag = 'mat'
                self.input = loadmat(input_dir)['data']
                self.target = loadmat(target_dir)['data']
                if len(self.input.shape) == 2:
                    self.input = self.input[0]
                if len(self.target.shape) == 2:
                    self.target = self.target[0]
                self.length = self.input.shape[0]
            elif "h5py" in input_dir:
                self.flag = "h5py"
                self.input = h5py.File(input_dir, 'r')['data']
                self.target = h5py.File(target_dir, 'r')['data']
                self.length = self.input.shape[0]
            else:
                assert 1 == 2, "the data type is not be supported at the moment"

        if self.flag == 'opencv':
            self.load_image = self.get_image_opencv
        elif self.flag == 'h5py':
            self.load_image = self.get_image_h5
        else:
            self.load_image = self.get_image_mat


    def __len__(self):
        return self.length

    def get_image_opencv(self, item):
        input = cv.imread(self.input[item], flags=-1)
        target = cv.imread(self.target[item], flags=-1)

        if input is None or target is None:
            input = Image.open(self.input[item])
            target = Image.open(self.target[item])
            input = np.array(input)
            target = np.array(target)
            input = cv.cvtColor(input, cv.COLOR_RGB2BGR)
            target = cv.cvtColor(target, cv.COLOR_RGB2BGR)

        if len(input.shape) == 3 and input.shape[2] == 3:
            if self.color == 'gray':
                input = cv.cvtColor(input, cv.COLOR_BGR2GRAY)
                target = cv.cvtColor(target, cv.COLOR_BGR2GRAY)
            else:
                input = cv.cvtColor(input, cv.COLOR_BGR2RGB)
                target = cv.cvtColor(target, cv.COLOR_BGR2RGB)

        return img_as_float32(input), img_as_float32(target)

    def get_image_h5(self, item):
        input = self.input[item]
        target = self.target[item]
        if len(input.shape) == 3 and input.shape[2] == 3:
            if self.color == 'gray':
                input = cv.cvtColor(input, cv.COLOR_RGB2GRAY)
                target = cv.cvtColor(target, cv.COLOR_RGB2GRAY)
        return img_as_float32(input), img_as_float32(target)

    def get_image_mat(self, item):
        input = self.input[item]
        target = self.target[item]
        return input, target

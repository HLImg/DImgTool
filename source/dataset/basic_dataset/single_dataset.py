# -*- coding: utf-8 -*-
# @Time    : 5/30/23 10:58 PM
# @File    : single_dataset.py.py
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


class BasicDataSetSingle(Dataset):
    def __init__(self, target_dir, color="gray", aug_rot=True, aug_flip=True):
        super(BasicDataSetSingle, self).__init__()
        self.color = color
        self.aug_rot = aug_rot
        self.aug_flip = aug_flip
        if os.path.isdir(target_dir):
            self.flag = 'opencv'
            self.input = data_util.get_image_path(directory=target_dir)
            self.length = len(self.input)
        elif 'mat' in target_dir:
            self.flag = 'mat'
            self.input = loadmat(target_dir)['data']
            if len(self.input.shape) == 2:
                self.target = self.input[0]
            self.length = self.input.shape[0]
        elif 'h5py' in target_dir:
            self.flag = 'h5py'
            self.input = h5py.File(target_dir, 'r')['data']
            self.length = self.input.shape[0]

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
        if input is None:
            input = Image.open(self.input[item])
            input = np.array(input)
            input = cv.cvtColor(input, cv.COLOR_RGB2BGR)

        if len(input.shape) == 3 and input.shape[2] == 3:
            if self.color == 'gray':
                input = cv.cvtColor(input, cv.COLOR_BGR2GRAY)
            else:
                input = cv.cvtColor(input, cv.COLOR_BGR2RGB)
        return img_as_float32(input)

    def get_image_h5(self, item):
        input = self.input[item]
        input = input.squeeze()
        if len(input.shape) == 3 and input.shape[2] == 3:
            if self.color == 'gray':
                input = cv.cvtColor(input, cv.COLOR_RGB2GRAY)
        return img_as_float32(input)

    def get_image_mat(self, item):
        input = self.input[item]
        return input

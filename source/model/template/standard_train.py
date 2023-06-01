# -*- coding: utf-8 -*-
# @Time    : 5/31/23 4:39 PM
# @File    : standard_train.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import torch
from source.utils.image.metric_util import TensorMetric
from source.model.basic_model.train_model import BasicModel

class Model(BasicModel):
    def __init__(self, opt, logger, main_dir, accelerator):
        super(Model, self).__init__(opt, logger, main_dir, accelerator)
        metric = self.opt["train"]["metric"]
        self.tensorMetric = TensorMetric(border=metric["border"], color=metric["color"])

    def __feed__(self, data_pair):
        self.optimizer[0].zero_grad()
        input, target = data_pair
        output = self.net[0](input)
        self.loss = self.criterion(output, target)
        # ==================================== #
        self.accelerator.backward(self.loss)
        # ==================================== #
        self.optimizer[0].step()
        self.scheduler[0].step()
        # TODO clip gradient

    def __eval__(self, data_pair):
        input, target = data_pair
        with torch.no_grad():
            output = self.net[0](input)
        # ======================================================================= #
        # gather data from multi-gpus
        output, target = self.accelerator.gather_for_metrics((output, target))
        # ======================================================================= #
        psnr = self.tensorMetric.calculate_psnr(output, target)
        return psnr





# -*- coding: utf-8 -*-
# @Time    : 5/31/23 12:58 PM
# @File    : __init__.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

from source.loss.basic_loss.image_loss import *
from source.loss.basic_loss.pixel_loss import *

class Loss:
    def __init__(self, opt):
        self.loss_info = opt["loss"]
        self.loss_type = opt["loss"]["type"]


    def __call__(self):
        if self.loss_type.lower() == "basic":
            from source.loss.basic_loss import  select_loss
        else:
            assert 1 == 2, f"the type-loss named {self.loss_name} is not exists."

        criterion = select_loss(self.loss_info)

        return criterion
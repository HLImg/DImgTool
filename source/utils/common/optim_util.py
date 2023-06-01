# -*- coding: utf-8 -*-
# @Time    : 5/30/23 9:48 PM
# @File    : optim_util.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import torch.optim as optim


def select_optimizer(name, param, net_param):
    param["params"] = net_param
    if name.lower() == "sgd":
        optimizer = optim.SGD(**param)
    elif name.lower() == "adam":
        optimizer = optim.Adam(**param)
    elif name.lower() == "adamw":
        optimizer = optim.AdamW(**param)
    else:
        assert 1 == 2, f"the name of optimizer <{name}> is incorrect"
    return optimizer


def select_scheduler(name, param, optimizer):
    param["optimizer"] = optimizer
    if name.lower() == "ExponentialLR".lower():
        scheduler = optim.lr_scheduler.ExponentialLR(**param)
    elif name.lower() == "StepLR".lower():
        scheduler = optim.lr_scheduler.StepLR(**param)
    elif name.lower() == "MultiStepLR".lower():
        scheduler = optim.lr_scheduler.MultiStepLR(**param)
    elif name.lower() == "CosineAnnealingLR".lower():
        scheduler = optim.lr_scheduler.CosineAnnealingLR(**param)
    else:
        assert 1 == 2, f"the name of scheduler < {name} > is incorrect"
    return scheduler

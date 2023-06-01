# -*- coding: utf-8 -*-
# @Time    : 5/31/23 11:39 AM
# @File    : pixel_loss.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import torch
import torch.nn as nn
import torch.nn.functional as F

class L1Loss(nn.Module):
    def __init__(self, weight=1., reduction="mean"):
        super(L1Loss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        loss = F.l1_loss(input, target, reduction=self.reduction)
        return self.weight * loss

class MSELoss(nn.Module):
    def __init__(self, weight=1., reduction="mean"):
        super(MSELoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        loss = F.mse_loss(input, target, reduction=self.reduction)
        return self.weight * loss


class CharbonnierLoss(nn.Module):
    def __init__(self, weight=1., eps=1e-3, reduction="mean"):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        diff = torch.add(target, -input)
        error = torch.sqrt(diff * diff + self.eps * self.eps)
        if self.reduction == "mean":
            loss = torch.mean(error)
        elif self.reduction == "sum":
            loss = torch.sum(error)
        else:
            loss = torch.mean(error)
        return self.weight * loss

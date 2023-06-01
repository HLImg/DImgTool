# -*- coding: utf-8 -*-
# @Time    : 5/30/23 9:55 PM
# @File    : train_util.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import os
import torch
import random
import numpy as np
import torch.nn as nn
from thop import profile
import torch.nn.init as init

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resume_state_test(checkpoint, net,  mode="all"):
    ckpt = torch.load(checkpoint, map_location=torch.device("cpu"))
    assert checkpoint is not None, 'checkpoint is None'
    if isinstance(net, list) and mode == "mine":
        for i in range(len(net)):
            if i == 0:
                net[0].load_state_dict(ckpt["nets"]["net_g"])
            else:
                net[i].load_state_dict(ckpt["nets"]["net_d" + str(i)])
    else:
        net[0].load_state_dict(ckpt[mode])
    return net


def resume_state_train(ckpt, nets, optimers=None, schedulers=None, mode="all"):
    ckpt = torch.load(ckpt, map_location=torch.device("cpu"))
    if mode == "all":
        next_iter = ckpt["iter"] + 1
        next_epoch = ckpt["epoch"] + 1
        best_score = ckpt["score"]
        # load net states
        for i in range(len(nets)):
            if i == 0:
                nets[0].load_state_dict(ckpt["nets"]["net_g"])
                if len(schedulers) > 0:
                    schedulers[i].load_state_dict(ckpt["schedulers"]["scheduler_g"])
                if len(optimers) > 0:
                    optimers[i].load_state_dict(ckpt["optimizers"]["optim_g"])
            else:
                nets[i].load_state_dict(ckpt["nets"]["net_d" + str(i)])
                if len(optimers) > 0:
                    optimers[i].load_state_dict(ckpt["optimizers"]["optim_d" + str(i)])
                if len(schedulers) > 0:
                    schedulers[i].load_state_dict(ckpt["schedulers"]["scheduler_d" + str(i)])
        return next_iter, next_epoch, nets, optimers, schedulers, best_score
    else:
        if isinstance(nets, list):
            for i in range(len(nets)):
                if i == 0:
                    nets[0].load_state_dict(ckpt["nets"]["net_g"])
                else:
                    nets[i].load_state_dict(ckpt["nets"]["net_d" + str(i)])
        else:
            nets.load_state_dict(ckpt["nets"])
        return nets


class Early_Stop:
    def __init__(self, logger, patience=50, verbose=False, delta=0, save_dir=None, mode='metric'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = 0
        self.early_stop = False
        self.val_loss_max = np.Inf
        self.val_metric_min = 0
        self.delta = delta
        self.save_dir = save_dir
        self.logger = logger
        if mode.lower() == 'metric':
            self.stop_mode = self.stop_metric
        else:
            self.stop_mode = self.stop_loss

    def stop_loss(self, iter, epoch, net_state, optimizer_state, scheduler_state, val_loss):
        score = -val_loss
        if self.best_score == 0:
            self.best_score = score
            self.save_state(iter, epoch, net_state, optimizer_state, scheduler_state, score=val_loss, valid_mode='loss')

        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.info(f'EarlyStopping counter : {self.counter} / {self.patience}.')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_state(iter, epoch, net_state, optimizer_state, scheduler_state, score=val_loss, valid_mode='loss')
            self.best_score = score
            self.counter = 0

    def stop_metric(self, iter, epoch, net_state, optimizer_state, scheduler_state, val_metric):
        score = val_metric
        if self.best_score == 0:
            self.best_score = score
            self.save_state(iter, epoch, net_state, optimizer_state, scheduler_state, score=val_metric, valid_mode='loss')

        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.logger.info(f'EarlyStopping counter : {self.counter} / {self.patience}.')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_state(iter, epoch, net_state, optimizer_state, scheduler_state, score=score, valid_mode='metric')
            self.best_score = score
            self.counter = 0

    def save_state(self, iter, epoch, net_state, optimizer_state, scheduler_state, score, valid_mode):
        if self.verbose:
            self.logger.info(
                f' epoch = {epoch}, validation ' + valid_mode + f' changed ({self.best_score:.6f} ---> {score:.6f}).')
        checkpoint = {
            'valid_mode': valid_mode,
            'iter': iter,
            'epoch': epoch,
            'nets': net_state,
            'optimizers': optimizer_state,
            'schedulers': scheduler_state,
            'score': score
        }
        save_name = os.path.join(self.save_dir, f'model_current_{str(epoch).zfill(4)}.pth')
        torch.save(obj=checkpoint, f=save_name)






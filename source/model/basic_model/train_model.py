# -*- coding: utf-8 -*-
# @Time    : 5/31/23 2:52 PM
# @File    : train_model.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import os
import torch
from source.loss import Loss
from source.network import Net
from source.dataset import DataSet
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler
from source.utils.common.initial_util import WeightInit
from source.utils.common.train_util import Early_Stop, resume_state_train
from source.utils.common.optim_util import select_optimizer, select_scheduler


class BasicModel:
    def __init__(self, opt, logger, main_dir, accelerator):
        """
        we use huggingface/accelerate to train.
        @param opt: type<dict> options
        @param logger: logger.info(<message>)
        @param main_dir: save directory
        @param accelerator: huggingface/accelerate
        note : the type of network, optimizer, scheduler is list
        """
        self.opt = opt
        self.logger = logger
        self.accelerator = accelerator
        num_gpu = len(opt["global_setting"]["device"].split(","))
        self.iteration = opt["train"]["iteration"]
        self.patience = opt["train"]["patience"]
        self.batch_per_gpu = opt["train"]["batch_per_gpu"]
        train_num_worker_per_gpu = opt["train"]["train_num_worker_per_gpu"]
        valid_batch_size = opt["train"]["valid_batch_size"]
        ckpt_save_dir = os.path.join(main_dir, opt['directory']['save_model'])
        runlog_save_dir = os.path.join(main_dir, opt['directory']['runlog'])

        self.writer = SummaryWriter(runlog_save_dir)
        self.dataset_train, self.dataset_valid = DataSet(self.opt)()
        self.loader_train = DataLoader(self.dataset_train, self.batch_per_gpu,
                                       shuffle=True, num_workers=train_num_worker_per_gpu)
        self.loader_valid = DataLoader(self.dataset_valid, valid_batch_size, shuffle=False,
                                       num_workers=2)

        self.logger.info(
            '# --------------------------------------------------------------------------------------------------------------------------#')
        self.logger.info(
            '#                    The dataLoader for vrain and validation has been loaded to the memory                                  #')
        self.logger.info(
            '# --------------------------------------------------------------------------------------------------------------------------#')
        self.early_stopper = Early_Stop(logger, patience=self.patience, verbose=True, delta=0, save_dir=ckpt_save_dir,
                                        mode=opt['train']['metric']['mode'])

        # TODO compile model
        self.net = Net(opt)()
        self.logger.info(
            '# --------------------------------------------------------------------------------------------------------------------------#')
        self.logger.info(
            '#                                  the model has been compiled by pytorch2.0                                                #')
        self.logger.info(
            '# --------------------------------------------------------------------------------------------------------------------------#')

        if self.opt['train']['resume'] == False and opt['train']['init']['state']:
            weight_init_function = WeightInit(name=opt['train']['init']['name'])
            self.net.apply(weight_init_function)

        self.optimizer = []
        self.scheduler = []

        self.epoch_begin = 1
        self.epoch_end = int(self.iteration // (self.dataset_train.length // self.batch_per_gpu * num_gpu))
        self.iter_begin = 1
        self.iter_end = self.iteration

        self.__setup_optimizer__()
        self.__setup_lr_scheduler__()
        if self.opt["train"]["resume"]["state"]:
            self.__resume__()

        self.criterion = Loss(opt)()
        

        # distributed training preparation is done by the accelerator

        for i in range(len(self.net)):
            self.net[i], self.optimizer[i] = self.accelerator.prepare(self.net[i], self.optimizer[i])
            if self.scheduler_state:
                self.scheduler[i] = self.accelerator.prepare(self.scheduler[i])
        self.loader_train, self.loader_valid = self.accelerator.prepare(self.loader_train, self.loader_valid)

    def __setup_lr_scheduler__(self):
        scheduler_info = self.opt["scheduler"]
        self.scheduler_state = scheduler_info["state"]
        if not self.scheduler_state:
            return
        num_d = len(scheduler_info) - 2
        self.scheduler.append(
            select_scheduler(scheduler_info["scheduler_g"]["name"],
                             scheduler_info["scheduler_g"]["param"], self.optimizer[0])
        )
        for i in range(num_d):
            self.scheduler.append(
                select_scheduler(scheduler_info["scheduler_d" + str(i + 1)]["name"],
                                 scheduler_info["scheduler_d" + str(i + 1)]["param"], self.optimizer[i + 1])
            )

    def __setup_optimizer__(self):
        optim_info = self.opt["optimizer"]
        num_d = len(optim_info) - 1
        self.optimizer.append(
            select_optimizer(optim_info["optim_g"]["name"],
                             optim_info["optim_g"]["param"], self.net[0].parameters()))
        for i in range(num_d):
            self.optimizer.append(
                select_optimizer(optim_info["optim_d" + str(i + 1)]["name"],
                                 optim_info["optim_d" + str(i + 1)]["param"], self.net[i + 1].parameters()))

    def __resume__(self):
        mode = self.opt["train"]["resume"]["mode"]
        checkpoint = self.opt["train"]["resume"]["ckpt"]
        if mode == "all":

            self.iter_begin, self.epoch_begin, self.net, \
                self.optimizer, self.scheduler, self.early_stopper.best_score = resume_state_train(checkpoint,
                                                                    self.net, self.optimizer, self.scheduler, mode=mode)
        else:
            self.net = resume_state_train(checkpoint, self.net)

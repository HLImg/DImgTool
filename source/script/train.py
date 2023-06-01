# -*- coding: utf-8 -*-
# @Time    : 5/31/23 5:04 PM
# @File    : train.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import os
import shutil
from tqdm import tqdm
from source.model.train import Model
from source.utils.common.log_util import Recorder, Logger


def train(model, opt, logger):
    best_metric = 0.0
    val_num = model.dataset_valid.__len__()
    for epoch in range(model.epoch_begin, model.epoch_end + 1):
        model.epoch = epoch
        loop_train = tqdm(model.loader_train, desc='training', disable=not model.accelerator.is_local_main_process)
        [net.train() for net in model.net]
        for _, data in enumerate(loop_train, 0):
            model.__feed__(data)
            model.iter_begin += 1
            loop_train.set_description(
                f"Main GPU : Epoch [{epoch} / {model.epoch_end}]")
            loop_train.set_postfix(loss=model.loss.item())

        if epoch % opt["train"]["valid_fre_epoch"] == 0:
            [net.eval()for net in model.net]
            metric = 0.0
            loop_valid = tqdm(model.loader_valid, desc="valid", disable=not model.accelerator.is_local_main_process)
            for _, data in enumerate(loop_valid, 0):
                metric += model.__eval__(data)
                loop_valid.set_description(
                f"Main GPU : Epoch [{epoch} / {model.epoch_end}]")
                loop_valid.set_postfix(BestPSNR=format(model.early_stopper.best_score , '.4f'), CurrentPSNR=format(metric / val_num, '.4f'))

            if model.accelerator.is_local_main_process:
                model.writer.add_scalar("valid_psnr", metric / val_num, epoch)
                # save training state
                net_states = {}
                optim_states = {}
                scheduler_states = {}
                for i in range(len(model.net)):
                    net_warp = model.accelerator.unwrap_model(model.net[i])
                    if i == 0:
                        net_states["net_g"] = net_warp.state_dict()
                        optim_states["optim_g"] = model.optimizer[i].state_dict()
                        if len(model.scheduler) > 0:
                            scheduler_states["scheduler_g"] = model.scheduler[i].state_dict()
                    else:
                        net_states["net_d" + str(i)] = net_warp.state_dict()
                        optim_states["optim_d" + str(i)] = model.optimizer[i].state_dict()
                        if len(model.scheduler) > 0:
                            scheduler_states["scheduler_d" + str(i)] = model.scheduler[i].state_dict()
                model.early_stopper.stop_metric(model.iter_begin, epoch,
                                                net_states, optim_states, scheduler_states, metric / val_num)
                if model.early_stopper.early_stop:
                    logger.info(f"early stop training, current epoch : {epoch}, iter : {model.iter_begin}")
                    break

    logger.info('# --------------------------------------------------------------------------------------------------------------------------#')
    logger.info('#                                                   Finish Trainin                                                          #')
    logger.info('# --------------------------------------------------------------------------------------------------------------------------#')


def in_let(opt, args, accelerator):
    recorder = Recorder(opt)
    recorder.__call__()
    _, yamlfile = os.path.split(args.yaml)
    shutil.copy(args.yaml, os.path.join(recorder.main_record, yamlfile))
    logger = Logger(log_dir=recorder.main_record)()
    logger.info(
        '# --------------------------------------------------------------------------------------------------------------------------#')
    logger.info(
        '#                                                   Start Training                                                          #')
    logger.info(
        '# --------------------------------------------------------------------------------------------------------------------------#')
    model = Model(opt, logger, recorder.main_record, accelerator)()
    train(model, opt, logger)

# -*- coding: utf-8 -*-
# @Time    : 6/1/23 12:30 PM
# @File    : test.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import os
import shutil

import numpy as np
from tqdm import tqdm
from scipy.io import savemat
from source.model.test import Model
from source.utils.common.log_util import Recorder, Logger

def test(model, logger):
    test_num = model.dataset_test.__len__()
    psnr, ssim, lpips = 0.0, 0.0, 0.0

    pbar = tqdm(model.dataloader, desc="test")
    for i, data in enumerate(pbar, 0):
        save_file = str(i).zfill(4) + ".png"
        info = model.__test__(save_file, data)
        psnr += info[0]
        ssim += info[1]
        lpips += info[2]
        pbar.set_description('testing ...')
        pbar.set_postfix(psnr=format(psnr / test_num, '.6f'), ssim=format(ssim / test_num, '.6f'),
                         lpips=format(lpips / test_num, '.6f'))
        logger.info(f'{save_file}   ----- > psnr = {info[0]: .6f},  ssim = {info[1]: .6f},  lpips = {info[2]: .6f}')

    if model.save["mat"]:
        savemat(model.restored_path, {"restored": np.array(model.restored, dtype=object)})


    logger.info(
        '# --------------------------------------------------------------------------------------------------------------------------#')
    logger.info(
        '#                                                   Finish Testing                                                          #')
    logger.info(
        '# --------------------------------------------------------------------------------------------------------------------------#')
    logger.info(
            f'Output-Target (average) : avg-psnr = {psnr / test_num : .6f}, avg-ssim = {ssim / test_num: .6f}, avg-lpips = {lpips / test_num : .6f}')


def in_let(opt, args):
    recorder = Recorder(opt)
    recorder.__call__()
    _, yamlfile = os.path.split(args.yaml)
    shutil.copy(args.yaml, os.path.join(recorder.main_record, yamlfile))
    logger = Logger(log_dir=recorder.main_record)()
    logger.info(
        '# --------------------------------------------------------------------------------------------------------------------------#')
    logger.info(
        '#                                                   Start Testing                                                          #')
    logger.info(
        '# --------------------------------------------------------------------------------------------------------------------------#')
    model = Model(opt, logger, recorder.main_record)()
    test(model, logger)
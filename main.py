# -*- coding: utf-8 -*-
# @Time    : 5/31/23 5:20 PM
# @File    : main.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn


import os
import argparse
import yaml
import source.script.test as test
import source.script.train as train
# TODO script for test
from source.utils.common.train_util import set_random_seed

# ========================================================== #
from accelerate import Accelerator
from accelerate.utils import set_seed
# ========================================================== #

parser = argparse.ArgumentParser()
parser.add_argument('--yaml', type=str, default='*.yaml')
args = parser.parse_args()

option = yaml.safe_load(open(args.yaml))
os.environ['CUDA_VISIBLE_DEVICES'] = option['global_setting']['device']

seed = option['global_setting']['seed']
set_seed(seed)
set_random_seed(seed=seed)

# accelerator = Accelerator()
# accelerator.print(f"device {str(accelerator.device)} is used")


if __name__ == '__main__':
    if option['global_setting']['action'].upper() == 'train'.upper():
        accelerator = Accelerator()
        accelerator.print(f"device {str(accelerator.device)} is used")
        train.in_let(opt=option, args=args, accelerator=accelerator)
    elif option['global_setting']['action'].upper() == 'test'.upper():
        test.in_let(option, args=args)

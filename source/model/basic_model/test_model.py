# -*- coding: utf-8 -*-
# @Time    : 5/31/23 9:20 PM
# @File    : test_model.py.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import os
from source.network import Net
from source.dataset import DataSet
from torch.utils.data import DataLoader
import source.utils.common.data_util as data_util
from source.utils.common.log_util import Recorder
from source.utils.image.metric_util import TensorMetric
from source.utils.common.train_util import resume_state_test

class BasicModel:
    def __init__(self, opt, logger, main_dir):
        self.opt = opt
        self.logger = logger
        self.main_dir = main_dir
        self.recoder = Recorder(opt)
        self.save = opt['test']['save']
        self.gpu = opt['test']['gpu']
        self.dpi = opt['test']['save']['dpi']
        self.metric_mode = opt['test']['metric']['mode']
        net_type = opt['test']['metric']['net_type']
        self.border = opt['test']['metric']['border']
        self.color = opt['test']['metric']['color']
        test_batch_size = opt['test']['batch_size']
        test_num_workers = opt['test']['num_worker']
        self.save_dir = os.path.join(main_dir, opt['directory']['vision'])
        self.output_dir = os.path.join(self.save_dir, "image")
        self.contrast_dir = os.path.join(self.save_dir, 'contrast')
        self.restored_path = os.path.join(self.save_dir, "restored.mat")

        os.mkdir(self.output_dir)
        os.mkdir(self.contrast_dir)

        self.net = Net(opt)()
        _, self.dataset_test = DataSet(opt)()

        self.dataloader = DataLoader(self.dataset_test, test_batch_size, shuffle=False,
                                     num_workers=test_num_workers)

        self.tensor_metric = TensorMetric(border=self.border, net_type=net_type, color=self.color)

        self.restored = []
        
        self.__resume__()

        if self.gpu:
            self.net = [net.cuda() for net in self.net]

        self.logger.info(
            '# --------------------------------------------------------------------------------------------------------------------------#')
        self.logger.info(
            '#                                                         Begin Testing                                                     #')
        self.logger.info(
            '# --------------------------------------------------------------------------------------------------------------------------#')

    def __resume__(self):
        resume = self.opt["test"]["resume"]
        mode = resume["mode"]
        self.net = resume_state_test(resume['ckpt'], net=self.net, mode=mode)


    def __save_single_tensor__(self, file, tensor):
        save_path = os.path.join(self.output_dir, file)
        data_util.save_tensor_single(tensor, save_path)


    def __save_constrast_tensor__(self, file, input, target, output):
        save_path = os.path.join(self.contrast_dir, file)
        data_util.save_tensor_contrast(input, target, output,
                                       border=self.border, channel=self.color,
                                       save_path=save_path, dpi=self.dpi)





# -*- coding: utf-8 -*-
# @Time    : 5/31/23 4:42 PM
# @File    : __init__.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

class Model:
    def __init__(self, opt, logger, main_dir, accelerator):
        self.model_info = opt["model"]
        self.task_name = opt["model"]["task"]
        self.opt = opt
        self.logger = logger
        self.main_dir = main_dir
        self.accelerator = accelerator

    def __call__(self):
        if self.task_name.lower() == "denoise":
            from source.model.train.denoise import select_model

        else:
            assert 1 == 2, f"the train model task named {self.task_name} is not exists"

        model = select_model(self.model_info, self.opt, self.logger,
                             self.main_dir, self.accelerator)
        return model

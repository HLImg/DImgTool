# -*- coding: utf-8 -*-
# @Time    : 6/1/23 12:34 PM
# @File    : __init__.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

class Model:
    def __init__(self, opt, logger, main_dir):
        self.model_info = opt["model"]
        self.task_name = opt["model"]["task"]
        self.opt = opt
        self.logger = logger
        self.main_dir = main_dir

    def __call__(self):
        if self.task_name.lower() == "denoise":
            from source.model.test.denoise import select_model
        else:
            assert 1 == 2, f"the test model task named {self.task_name} is not exists"

        model = select_model(self.model_info, self.opt, self.logger, self.main_dir)
        return model

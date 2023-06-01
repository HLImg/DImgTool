# -*- coding: utf-8 -*-
# @Time    : 5/30/23 11:06 PM
# @File    : __init__.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

class DataSet:
    def __init__(self, opt):
        self.dataset_info = opt["dataset"]
        self.task_name = opt["dataset"]["task"]

    def __call__(self):
        if self.task_name.lower() == "denoise":
            from source.dataset.denoise import select_dataset

        else:
            assert 1 == 2, f"the task named {self.task_name} is not exists."

        train_dataset, test_dataset = select_dataset(self.dataset_info)
        return train_dataset, test_dataset



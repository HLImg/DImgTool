# -*- coding: utf-8 -*-
# @Time    : 5/31/23 2:31 PM
# @File    : __init__.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

class Net:
    def __init__(self, opt):
        self.net_info = opt["network"]
        self.task_name = opt["network"]["task"]

    def __call__(self):
        """
        Since GAN consists of a generator and a discriminator, we set the return type as a list.
        @return: [net_g, net_d1, net_d2 ]
        """
        if self.task_name.lower() == "denoise":
            from source.network.denoise import select_network

        else:
            assert 1 == 2, f"the task named {self.task_name} is not exists."

        network = select_network(self.net_info)
        return network
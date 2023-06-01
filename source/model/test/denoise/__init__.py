# -*- coding: utf-8 -*-
# @Time    : 6/1/23 12:33 PM
# @File    : __init__.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

def select_model(model_info, opt, logger, main_dir):
    name = model_info["name"]
    if name.lower() == "standard":
        from source.model.template.standard_test import Model

    model = Model(opt, logger, main_dir)
    return model

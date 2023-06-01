# -*- coding: utf-8 -*-
# @Time    : 5/31/23 4:42 PM
# @File    : __init__.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

def select_model(model_info, opt, logger, main_dir, accelerator):
    name = model_info["name"]
    if name.lower() == "standard":
        from source.model.template.standard_train import Model

    model = Model(opt, logger, main_dir, accelerator)
    return model
# -*- coding: utf-8 -*-
# @Time    : 5/30/23 11:07 PM
# @File    : __init__.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

def select_dataset(dataset_info):
    name = dataset_info["name"]
    params = dataset_info["params"]
    train_dataset, test_dataset = (None, None)

    if name.lower() == "synthetic":
        from source.dataset.denoise.synthetic import TrainDataSet as Train
        from source.dataset.denoise.synthetic import TestDataSet as Test

    elif name.lower() == "pair_load":
        from source.dataset.template.pair_load import TrainDataSet as Train
        from source.dataset.template.pair_load import TestDataSet as Test
    
    elif name.lower() == "syn_pair":
        from source.dataset.denoise.synthetic import TrainDataSet as Train
        from source.dataset.template.pair_load import TestDataSet as Test

    else:
        assert 1 == 2, f"the dataset named {name} is not exists"

    if "train" in params.keys():
        train_dataset = Train(**params["train"])
    if "test" in params.keys():
        test_dataset = Test(**params["test"])

    return train_dataset, test_dataset

# -*- coding: utf-8 -*-
# @Time    : 5/31/23 2:23 PM
# @File    : __init__.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn


from source.network.denoise.dncnn.model_arch import DnCNN
from source.network.denoise.mirnetv2.model_arch import MIRNet_v2


def select_network(net_info):
    network = []
    num_d = len(net_info) - 2
    network.append(choose(net_info["net_g"]["name"], net_info["net_g"]["param"]))
    for i in range(num_d):
        network.append(choose(net_info["net_d" + str(i + 1)]["name"],
                              net_info["net_d" + str(i + 1)]["param"]))
    assert len(network) > 0, f"failed on selected network "
    return network

def choose(name, param):
    if name.lower() == "dncnn":
        model = DnCNN(**param)
    elif name.lower() == "mirnetv2":
        model = MIRNet_v2(**param)
    
    else:
        assert 1 == 2, f"the network named {name} is not exists."

    return model
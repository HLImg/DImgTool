# -*- coding: utf-8 -*-
# @Time    : 5/31/23 9:46 PM
# @File    : standard_test.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import torch
from source.model.basic_model.test_model import BasicModel

class Model(BasicModel):
    def __init__(self, opt, logger, main_dir):
        super(Model, self).__init__(opt, logger, main_dir)

    def __test__(self, file, data_pir):
        data_pir = data_pir[:2]
        with torch.no_grad():
            input, target = data_pir
            if self.gpu:
                input, target = [x.cuda() for x in data_pir]
            output = self.net[0](input)

        if self.save["image"]:
            self.__save_single_tensor__(file, output)
        if self.save["contrast"]:
            self.__save_constrast_tensor__(file, input, target, output)
        if self.save["mat"]:
            restored = torch.clamp(output, 0, 1).cpu().detach().permute(0, 2, 3, 1).numpy()
            for i in range(restored.shape[0]):
                self.restored.append(restored[i].squeeze())

        psnr, ssim, lpips = (-1, -1, -1)
        if self.metric_mode["psnr"]:
            psnr = self.tensor_metric.calculate_psnr(output, target)
        if self.metric_mode["ssim"]:
            ssim = self.tensor_metric.calculate_ssim(output, target)
        if self.metric_mode["lpips"]:
            lpips = self.tensor_metric.calculate_lpips(output, target)
        return psnr, ssim, lpips

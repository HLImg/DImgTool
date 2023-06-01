# -*- coding: utf-8 -*-
# @Time    : 5/31/23 1:38 PM
# @File    : __init__.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

def select_loss(loss_info):
    pixel = loss_info.get("pixel", False)
    image = loss_info.get("image", False)
    pixel_loss, image_loss = None, None

    if pixel:
        pixel_name = pixel["name"]
        if pixel_name.lower() == "l1":
            from source.loss.basic_loss.pixel_loss import L1Loss as PixelLoss
        elif pixel_name.lower() == "mse":
            from source.loss.basic_loss.pixel_loss import MSELoss as PixelLoss
        elif pixel_name.lower() == "char":
            from source.loss.basic_loss.pixel_loss import CharbonnierLoss as PixelLoss
        else:
            assert 1 == 2, f"the pixel-loss named {pixel_name} is not exists."
        pixel_loss = PixelLoss(**pixel["param"])

    if image:

        image_name = image["name"]
        if image_name.lower() == "perceptual":
            from source.loss.basic_loss.image_loss import PerceptualLoss as ImageLoss
        elif image_name.lower() == "edge":
            from source.loss.basic_loss.image_loss import EdgeLoss as ImageLoss
        else:
            assert 1 == 2, f"the image-loss named {image_name} is not exists"
        image_loss = ImageLoss(**image["param"])

    if pixel_loss is not None and image_loss is not None:
        loss = lambda input, target: image_loss(input, target) + pixel_loss(input, target)
        return loss
    if pixel_loss is not None:
        loss = lambda input, target: pixel_loss(input, target)
        return loss
    if image_loss is not None:
        loss = lambda input, target: image_loss(input, target)
        return loss
    assert 1 == 2, "pixel_loss and image_loss are all None"



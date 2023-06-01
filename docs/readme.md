#### Simple Introduction

This is a **training template** for beginners in image processing or low-level vision on quickly train their own deep networks for image restoration. It is built on PyTorch ans uses HuggingFace's accelerate tool for distributed training. This is a *simple* template. For professionals, I recommend using BasicSR.

In this template, we need to build our own jobs from modules such as **dataset**, **loss**, **network**, and **model**, and add their initialization api to **__init__.py** according to the task (e.g. denoising, super-resolution).
Finally, we can build the configuration in a yaml file.

#### How to run


```shell
# training
accelerate launch --config_file config.yaml main.py --yaml z_local/train.yaml
# test 

```
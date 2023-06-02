### Simple Introduction

This is a **training template** for beginners in image processing or low-level vision on quickly train their own deep networks for image restoration. It is built on PyTorch and uses HuggingFace's accelerate tool for distributed training. This is a *simple* template. For professionals, I recommend using BasicSR.

In this template, we need to build our own jobs from modules such as **dataset**, **loss**, **network**, and **model**, and add their initialization api to **__init__.py** according to the task (e.g. denoising, super-resolution).
Finally, we can build the configuration in a yaml file.

#### How to run


```shell
# training: ddp training
accelerate launch --config_file config.yaml main.py --yaml z_local/train.yaml
# test : single gpu or cpu
python main.py --yaml z_local/train.yaml
```

#### ToDo List
- [ ] Training Template
	- [x] Distributed training by accelerate.
	- [x] Support GAN training, can add a generative network and multiple discriminators
	- [x] Support *train* data in **h5py** format.
	- [x] Support *valid / test* data in **mat** format.
	- [ ] **Further testing the effectiveness and accuracy of GAN training**.
	
- [ ] Model reproduction based on **Training Template**
	- [ ] Image Denoising (SIDD)
		- [ ] MIRNet-V2
		- [ ] MPRNet
		- [ ] NAFNet
	- [ ] Image Super-Resolution (Set5, Set14, Urban100)
		- [ ] SwinIR
		- [ ] RCAN
		- [ ] BSRN
	- [ ] Image Deblurring (GoPro)
		- [ ] MPRNet
		- [ ] NAFNet
	

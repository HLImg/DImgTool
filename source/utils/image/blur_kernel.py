# -*- coding: utf-8 -*-
# @Time    : 5/30/23 10:26 PM
# @File    : blur_kernel.py
# @Author  : Hao Liang
# @Email   : lianghao@whu.edu.cn

import numpy as np
from scipy import signal


class GaussianBlurringKernel:
    """
    refer the BasicSR of Wang X. T.
    """

    def sigma_matrix_2d(self, sigma_x, sigma_y, theta):
        # Calculate the rotated sigma matrix (2-dim matrix)
        d_matrix = np.array([[sigma_x ** 2, 0], [0, sigma_y ** 2]])
        u_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        matrix = np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))
        return matrix

    def mesh_grid(self, kernel_size):
        # Generate the mesh grid , centering at zero
        ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        xy = np.hstack((xx.reshape((kernel_size * kernel_size, 1)), yy.reshape(kernel_size * kernel_size, 1))).reshape(
            kernel_size, kernel_size, 2
        )
        return xy, xx, yy

    def pdf2(self, sigma_matrix, grid):
        # Calculate PDF of the bevariate Gaussian d/lolistribution
        inverse_sigma = np.linalg.inv(sigma_matrix)
        kernel = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))
        return kernel

    @staticmethod
    def bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid=None, isotropic=True):
        # Generate a bivariate isotropic or anisotropic Gaussian kernel.
        if grid is None:
            grid, _, _ = GaussianBlurringKernel.mesh_grid(self=GaussianBlurringKernel, kernel_size=kernel_size)
        if isotropic:
            sigma_matrix = np.array([[sig_x ** 2, 0], [0, sig_x ** 2]])
        else:
            sigma_matrix = GaussianBlurringKernel.sigma_matrix2(self=GaussianBlurringKernel, sig_x=sig_x, sig_y=sig_y,
                                                                theta=theta)
        kernel = GaussianBlurringKernel.pdf2(self=GaussianBlurringKernel, sigma_matrix=sigma_matrix, grid=grid)
        kernel = kernel / np.sum(kernel)
        return kernel

    @staticmethod
    def random_bivariate_Gaussian(kernel_size, sigma_x_range, sigma_y_range, rotation_range, isotropic=True):
        # Randomly generate bivariate isotropic or anisotropic Gaussian kernels
        assert sigma_x_range[0] < sigma_x_range[1], 'sigma_x : minimum >= maximum'
        sigma_x = np.random.uniform(low=sigma_x_range[0], high=sigma_x_range[1])
        if isotropic is False:
            assert sigma_y_range[0] < sigma_y_range[1], 'sigma_y : minimum >= maximum'
            assert rotation_range[0] < rotation_range[1], 'rotation : minimum >= maximum'
            sigma_y = np.random.uniform(low=sigma_y_range[0], high=sigma_y_range[1])
            rotation = np.random.uniform(low=rotation_range[0], high=rotation_range[1])
        else:
            sigma_y = sigma_x
            rotation = 0
        kernel = GaussianBlurringKernel.bivariate_Gaussian(kernel_size=kernel_size, sig_x=sigma_x, sig_y=sigma_y,
                                                           theta=rotation, isotropic=isotropic)
        kernel = kernel / np.sum(kernel)
        return kernel


def motion_blur_deblurgan(image, kernel, channel):
    k_h = kernel.shape[:1]
    h = image.shape[:1]
    assert h >= k_h, 'resolution of image should be higher than kernel'
    kernel = np.pad(kernel, (h - k_h) // 2, 'constant')
    if channel == 'gray':
        blurred = np.array(signal.fftconvolve(image, kernel, 'same'))
    else:
        blurred[:, :, 0] = np.array(signal.fftconvolve(image[:, :, 0], kernel, 'same'))
        blurred[:, :, 1] = np.array(signal.fftconvolve(image[:, :, 1], kernel, 'same'))
        blurred[:, :, 2] = np.array(signal.fftconvolve(image[:, :, 2], kernel, 'same'))
    return np.abs(blurred)


def psf2oft(kernel, shape):
    """
    the code from "Photon Limited Non-Blind Deblurring Using Algorithm Unrolling"
    Args:
        kernel (_type_): motion blurring kernel
        shape (_type_):the shape of image
    """
    psf = np.zeros(shape, dtype=np.float32)
    center = np.shape(kernel)[0] // 2 + 1
    psf[:center, :center] = kernel[(center - 1):, (center - 1):]
    psf[:center, -(center - 1):] = kernel[(center - 1):, :(center - 1)]
    psf[-(center - 1):, :center] = kernel[:(center - 1), (center - 1):]
    psf[-(center - 1):, -(center - 1):] = kernel[:(center - 1), :(center - 1)]
    otf = np.fft.fft2(psf, shape)
    return psf, otf


def motion_blur_p4ip(image, kernel, channel='gray'):
    _, k_fft = psf2oft(kernel, shape=image.shape[:2])
    if channel == 'gray':
        blurred = np.real(np.fft.ifft2(image) * k_fft)
    else:
        blurred[:, :, 0] = np.real(np.fft.ifft2(np.fft.fft2(image[:, :, 0]) * k_fft))
        blurred[:, :, 1] = np.real(np.fft.ifft2(np.fft.fft2(image[:, :, 1]) * k_fft))
        blurred[:, :, 2] = np.real(np.fft.ifft2(np.fft.fft2(image[:, :, 2]) * k_fft))
    return np.abs(blurred)
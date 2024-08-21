import numpy as np
from astropy.convolution import RickerWavelet2DKernel


def choose_kernel(kernel_type, size=3):
    # print(size)
    if kernel_type == 'gaussian':
        kernel = gaussian_kernel(size=size, sigma=size / 3)
        return kernel
    if kernel_type == 'mexican_hat':
        kernel = ricker_kernel(size=size)
        return kernel


def gaussian_kernel(size, sigma=1.):
    """
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def ricker_kernel(size=3, sigma=1.):
    return RickerWavelet2DKernel(width=sigma, x_size=size, y_size=size)
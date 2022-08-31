import math
import numpy as np


def mean_kernel(shape: tuple, val=None):
    if(val):
        return np.full(shape, val)
    else:
        return np.full(shape, float(1) / float(shape[0] * shape[1]))


def prewitt_kernel_x():
    return np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]).astype(np.double)


def prewitt_kernel_y():
    return np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]).astype(np.double)


def sobel_kernel_x():
    return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.double)


def sobel_kernel_y():
    return np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.double)


def gaussian_kernel(sigma):
    # Ensure filter always > 0
    kernel_size = 2 * int(4 * sigma + 0.5) + 1
    kernel = np.zeros((kernel_size, kernel_size), np.float32)

    m = math.floor(kernel_size / 2)
    n = math.floor(kernel_size / 2)

    for x in range(-m, m+1):
        for y in range(-n, n+1):
            x1 = 2*np.pi*(sigma**2)
            x2 = np.exp(-(x**2 + y**2)/(2 * sigma**2))
            kernel[x+m, y+n] = (1/x1)*x2

    return kernel


def visualize_kernel(kernel):
    import matplotlib.pyplot as plt

    ax = plt.axes(projection="3d")

    x = np.arange(kernel.shape[0])
    y = np.arange(kernel.shape[1])

    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, kernel, rstride=1, cstride=1,
                    cmap='winter', edgecolor='none')

    plt.show()

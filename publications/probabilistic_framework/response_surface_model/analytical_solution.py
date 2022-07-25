import numpy as np
from scipy.special import erf
from matplotlib import pyplot as plt


def Ft(x1, x2):
    return np.array([1, x1, x2, x1*x2, x1**2, x2**2])

def contours(theta, x1, x2):
    sigma_theta = 0.05 * np.eye(6)
    first = erf((3-Ft(x1, x2) @ theta)/(np.sqrt(2*(Ft(x1, x2)[:, None].T @ sigma_theta @ Ft(x1, x2)[:, None])[0, 0])))
    second = erf((1.85-Ft(x1, x2) @ theta)/(np.sqrt(2*(Ft(x1, x2)[:, None].T @ sigma_theta @ Ft(x1, x2)[:, None])[0, 0])))

    return first - second

if __name__ == '__main__':
    reso = 101
    theta = [2, 1, 1, 1, 2, 2]
    x = np.linspace(-1, 1, reso)
    y = np.linspace(-1, 1, reso)
    z = []
    for x_i in x:
        for y_i in y:
            z.append(contours(theta, x_i, y_i))
    x, y = np.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()
    levels = [0, 2 * 0.85, 2*1.00]
    plt.tricontour(x, y, z, levels=levels, colors=["black", "blue"], zorder=0)
    plt.show()

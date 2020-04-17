from numpy import ndarray
from .gaussmf import gaussmf


class Gauss2MF:
    def __init__(self, mean1: float, sigma1: float, mean2: float, sigma2: float):
        self.mean1 = mean1  # c
        self.mean2 = mean2
        self.sigma1 = sigma1  # sigma
        self.sigma2 = sigma2


def gauss2mf(x: ndarray, params: Gauss2MF) -> ndarray:
    xa = x
    for k in range(len(x)):
        if xa[k] <= params.mean1:
            xa[k] = 1
    x0 = gaussmf(x, params.mean1, params.sigma1)

    xb = x
    for k in range(len(x)):
        if xb[k] > params.mean2:
            xb[k] = 1
    x1 = gaussmf(x, params.mean2, params.sigma2)

    return x0 * x1

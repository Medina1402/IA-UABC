import matplotlib.pyplot as plt
from numpy import *


def plotperf(data: ndarray):
    x = arange(0, size(data), 1)
    coef = polyfit(x, data, 1)
    poly = poly1d(coef)

    plt.plot(x, data, x, poly(x), '--k')
    plt.show()

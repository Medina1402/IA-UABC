import matplotlib.pyplot as plt
from numpy import *


def plotregression(x: ndarray, y: ndarray):
    coef = polyfit(x.ravel(), y.ravel(), 1)
    poly = poly1d(coef)
    plt.grid()
    plt.plot(x, y, 'yo', x, poly(x), '--k')
    plt.show()

from numpy import transpose, array, ndarray

from .genfis1mamcost1fls import *
from .genparamt1fls import *
from .typedata import FIS
from ..matrix import reshapeVecToMatrix


def fnnget(fis: FIS) -> (ndarray, ndarray, ndarray):
    n = len(fis.input)  # entrada
    m = len(fis.output)  # salidas
    r = len(fis.rule)  # reglas

    SIGMA = []
    CENTER = []
    THETA = []
    for k in range(r):
        index = fis.rule[k].antecedent
        sigma = []
        center = []

        for i in range(n):
            param = fis.input[i].mf[index[i]].params
            sigma.append(param[1])
            center.append(param[1])
        SIGMA.append([sigma])
        CENTER.append([center])

    for j in range(m):
        xn = fis.output[j].mf[j].params
        out_Params = reshapeVecToMatrix(array([xn]), 1, r)
        THETA.append([transpose(out_Params)])

    return array(SIGMA), array(CENTER), array(THETA)


def fnnsetm(fis: FIS, SIGMA: ndarray, CENTER: ndarray, THETA: ndarray) -> FIS:
    n = len(fis.input)  # entradas
    m = len(fis.output)  # salidas
    r = len(fis.rule)  # reglas

    for i in range(n):
        for k in range(r):
            fis.input[i].mf[k].params = [SIGMA[k][i], CENTER[k][i]]

    for j in range(m):
        theta = reshapeVecToMatrix(THETA[:][j], 1, r)
        for k in range(r):
            fis.output[j].mf[k].params = theta[k][:]

    return fis

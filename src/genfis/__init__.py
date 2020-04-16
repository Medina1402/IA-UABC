from numpy import transpose, array

from .genfis1mamcost1fls import *
from .genparamt1fls import *
from .typedata import FIS
from ..mamdani.typedata import DesignParams
from ..matrix import reshapeVecToMatrix


def fnnget(fis: FIS):
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
        xn = fis.output[j].mf.params
        out_Params = reshapeVecToMatrix(array([xn]), 1, r)
        THETA.append([transpose(out_Params)])

    return DesignParams(array(SIGMA), array(CENTER), array(THETA))

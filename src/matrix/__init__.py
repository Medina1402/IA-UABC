import numpy

from .colMatrixAugment import *
from .reshapeVecToMatrix import *
from .vectorizeMatrixTocolVec import *


def normalize(A: ndarray) -> float:
    """
    Normalizacion Euclidanea
    *************************
    La matriz se convierte en un vector y cada elemento es elevado al cuadrado
    se suman y se obtiene la raiz del total, dicha raiz es nuestra matriz normalizada

    >> A = [[1, 2, 3], [3, 2, 1]]
    >> normalize(A)
    >> 2.23606797749979

    :param A:
    :return:
    """
    value = 0
    V = ravel(A)**2
    for k in range(len(V)):
        value += V[k]
    return value**0.5

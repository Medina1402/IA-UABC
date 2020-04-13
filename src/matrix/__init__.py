from .colMatrixAugment import *
from .reshapeVecToMatrix import *
from .vectorizeMatrixTocolVec import *


def normalize(A: ndarray):
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
    V = ravel(A)
    for k in range(len(A)):
        value += V[k]**2
    return value ** .5

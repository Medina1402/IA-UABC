from numpy import ndarray

from src.mamdani.typedata import DesignParams
from src.matrix import reshapeVecToMatrix


def reshapeParam(designParam: DesignParams, V: ndarray) -> DesignParams:
    """
    :param designParam:
    :param V: array (2rn+rm)x1
    :return: designParam
    """

    r, n = designParam.sigma.shape # reglas, entradas
    _, m = designParam.theta.shape # salidas

    np1 = 2 * r * n # antecedentes
    np2 = m * r # consecuentes
    np3 = np1 + np2 # total de parametros

    V1 = V[:np1]
    VT = V[np1: np3]
    VS = V1[:int(np1/2)]
    VC = V1[int(np1/2):np1]

    SIGMA = reshapeVecToMatrix(VS, r, n)
    CENTER = reshapeVecToMatrix(VC, r, n)
    THETA = reshapeVecToMatrix(VT, r, m)

    return DesignParams(SIGMA, CENTER, THETA)
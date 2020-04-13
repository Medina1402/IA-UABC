from numpy import ndarray
from src.mamdani import DesignParams
from src.matrix import reshapeVecToMatrix


def reshapeParam(designParam: DesignParams, V: ndarray) -> DesignParams:
    """
    :param designParam:
    :param V: array (2rn+rm)x1
    :return:
    """

    r, n = designParam.sigma
    _, m = designParam.theta

    VS = V[:r * n]
    VM = V[r * n: r * n * 2]
    VC = V[r * n * 2:]

    SIGMA = reshapeVecToMatrix(VS, r, n)
    CENTER = reshapeVecToMatrix(VM, r, n)
    THETA = reshapeVecToMatrix(VC, r, m)

    designParam2 = DesignParams(SIGMA, CENTER, THETA)
    return designParam2

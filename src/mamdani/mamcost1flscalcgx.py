from numpy import ndarray, zeros, multiply, transpose

from src.mamdani.typedata import DesignParams
from src.matrix import colMatrixAugment, vectorizeMatrixTocolVec, normalize


def mamcost1flscalcgx(designParam: DesignParams, X: ndarray, T: ndarray, Y: ndarray, E: ndarray, PHI: ndarray) \
        -> (ndarray, float, ndarray):
    """
    :param designParam:
    :param X: array qxn
    :param T: array qxm
    :param Y: array qxm
    :param E: array qxm
    :param PHI: array qxr
    :return: gX array (2rn+rm)x1, normgX float, Jew array mqx(2rn+rm)
    """
    q, n = X.shape
    _, m = T.shape
    r, _ = designParam.sigma.shape

    SIGMA = designParam.sigma
    CENTER = designParam.center
    THETA = designParam.theta

    de_ds = zeros([q * m, r * n])
    de_dm = zeros([q * m, r * n])
    de_dc = zeros([q * m, r * m])

    for j in range(m):
        for p in range(q):
            row = (j-1) * q + p
            for i in range(n):
                for k in range(r):
                    col = (i-1) * r + k
                    temp1 = (THETA[k][j] - Y[p][j]) * PHI[p][k]
                    temps = (X[p][i] - CENTER[k][i]) ** 2 / SIGMA[k][i] ** 3
                    de_ds[row][col] = -temp1 * temps
                    tmpm = (X[p][i] - CENTER[k][i]) / SIGMA[k][i] ** 2
                    de_dm[row][col] = -temp1 * tmpm

            for k in range(r):
                col = (j-1) * r + k
                de_dc[row][col] = -PHI[p][k] * -1
    Jew = colMatrixAugment(de_ds, de_dm)
    Jew = colMatrixAugment(Jew, de_dc)
    ew = vectorizeMatrixTocolVec(E)
    gX = 2 * multiply(ew, transpose(Jew))
    normgX: float = normalize(gX)

    return gX, normgX, Jew

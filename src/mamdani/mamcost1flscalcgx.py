from numpy import ndarray, zeros, matmul, dot

from src.mamdani.typedata import DesignParams
from src.matrix import colMatrixAugment, normalize


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
    _, r = PHI.shape

    SIGMA = designParam.sigma
    CENTER = designParam.center
    THETA = designParam.theta

    de_ds = zeros([q * m, r * n])
    de_dm = zeros([q * m, r * n])
    de_dc = zeros([q * m, r * m])

    for j in range(m):
        for p in range(q):
            row = j * q + p
            for i in range(n):
                for k in range(r):
                    col = i * r + k
                    temp1 = (THETA[k][j] - Y[p][j]) * PHI[p][k]
                    temps = (X[p][i] - CENTER[k][i])**2 / SIGMA[k][i]**3
                    de_ds[row][col] = -temp1 * temps
                    tmpm = (X[p][i] - CENTER[k][i]) / SIGMA[k][i]**2
                    de_dm[row][col] = -temp1 * tmpm

            for k in range(r):
                col = j * r + k
                de_dc[row][col] = -PHI[p][k]

    Jew = colMatrixAugment(de_ds, de_dm)
    Jew = colMatrixAugment(Jew, de_dc)
    ew: ndarray = E.ravel(order="F")  # vectorizeMatrix
    gX: ndarray = 2 * dot(Jew.transpose(), ew)
    normgX = normalize(gX)

    return gX, normgX, Jew, ew

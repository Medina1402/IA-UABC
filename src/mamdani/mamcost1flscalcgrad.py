from numpy import ndarray, zeros, array

from src.mamdani.typedata import DesignParams
from src.matrix import normalize


def mamcost1flscalcgrad(designParam: DesignParams, X: ndarray, T: ndarray, Y: ndarray, E: ndarray, PHI: ndarray) -> (ndarray, float):
    """
    :param designParam:
    :param X: array qxn
    :param T: array qxm
    :param Y: array qxm
    :param E: array qxm
    :param PHI: array qxr
    :return: gX array (2rn+rm)x1, normgX float
    """
    q, n = X.shape
    _, m = T.shape
    _, r = PHI.shape

    SIGMA = designParam.sigma
    CENTER = designParam.center
    THETA = designParam.theta

    dE_dS = zeros([r, n])
    dE_dM = zeros([r, n])
    dE_dC = zeros([r, n])

    for p in range(q):
        dEp_ds = zeros([r, n])
        dEp_dm = zeros([r, n])
        dEp_dc = zeros([r, m])

        for i in range(n):
            for k in range(r):
                A = PHI[p][k] * ((X[p][i] - CENTER[k][i]) / SIGMA[k][i] ** 2)
                B = A * (X[p][i] - CENTER[k][i]) / SIGMA[k][i]
                S = 0
                for j in range(m):
                    S = S - 2 * E[p][j] * (THETA[k][j] - Y[p])[j]
                dEp_dm[k][i] = A * S
                dEp_ds[k][i] = B * S

        for k in range(r):
            for j in range(m):
                dEp_dc[k][j] = -2 * E[p][j] * PHI[p][k]

        dE_dS += dEp_ds
        dE_dM += dEp_dm
        dE_dC += dEp_dc

    gX: ndarray = array([dE_dS[:], dE_dM[:], dE_dC[:]])
    normgX: float = normalize(gX)
    return gX, normgX

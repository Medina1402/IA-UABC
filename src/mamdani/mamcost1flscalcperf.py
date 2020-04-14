from numpy import ndarray, zeros, math
from src.mamdani.typedata import DesignParams


def mamcost1flscalcperf(designParam: DesignParams, X: ndarray, T: ndarray) -> (float, ndarray, ndarray, float, ndarray):
    """
    :param designParam:
    :param X: array qxn
    :param T: array qxm
    :return: SEE R, Y array, E array, PHI array, ALPHA array
    """
    SIGMA = designParam.sigma
    CENTER = designParam.center
    THETA = designParam.theta

    q, n = X.shape
    _, m = T.shape
    r, _ = CENTER.shape

    ALPHA = zeros([q, r])
    PHI = zeros([q, r])
    E = zeros([q, m])
    Y = zeros([q, m])

    for p in range(q):
        sum_alpha = 0
        for k in range(r):
            suma = 0
            for i in range(n):
                suma += ((X[p][i] - CENTER[k][i]) / SIGMA[k][i]) ** 2

            ALPHA[p][k] = math.e ** (suma * -0.5)
            sum_alpha += ALPHA[p][k]

        for k in range(r):
            PHI[p][k] = ALPHA[p][k] / sum_alpha

        for j in range(m):
            Y[p][j]: float = 0
            for k in range(r):
                Y[p][j] += THETA[k][j] * PHI[p][k]

    SEE: float = 0
    for p in range(q):
        for j in range(m):
            E[p][j] = T[p][j] - Y[p][j]
            SEE += E[p][j]**2

    return SEE, Y, E, PHI, ALPHA

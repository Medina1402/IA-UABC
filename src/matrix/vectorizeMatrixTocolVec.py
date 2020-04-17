from numpy import ndarray, ravel


def vectorizeMatrixTocolVec(A: ndarray) -> ndarray:
    """
    :param A: array nxm,
    :return: array nmx1
    """
    return ravel(A)

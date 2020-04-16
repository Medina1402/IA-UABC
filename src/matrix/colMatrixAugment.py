from numpy import column_stack, ndarray


def colMatrixAugment(A: ndarray, B: ndarray) -> ndarray:
    """
    Se agregara B seguido de las columnas de A
    >> A = [[1, 2], [3, 4]]
    >> B = [[2, 1, 8], [4, 3, 8]]
    >> colMatrixAugment(A, B)
    >> [[1, 2, 2, 1, 8], [3, 4, 4, 3, 8]]

    :param A: array nxm
    :param B: array nxL
    :return: array nx(m+L)
    """

    M = column_stack([A, B])
    return M

from numpy import ndarray


def reshapeVecToMatrix(V: ndarray, n: int, m: int) -> ndarray:
    """
    Convierte un arreglo o tupla unidimencional (vector) en uno multidimensional.
    >> A = [1, 2, 3, 4]
    >> reshapeVecToMatrix(A, 2, 2)
    >> [[1, 2], [3, 4]]

    :param V: array nmx1
    :param n: num rows
    :param m: num columns
    :return: array nxm
    """

    A = V.reshape(n, m)
    return A

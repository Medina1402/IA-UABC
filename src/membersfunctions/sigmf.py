from numpy import ndarray, array, e


# ==========================================================
# Member Functions ** SIGMOIDAL **
# ==========================================================
class Sig2MF:
    def __init__(self, a1: float, c1: float, a2: float, c2: float):
        self.a1 = a1
        self.c1 = c1
        self.a2 = a2
        self.c2 = c2


class SigMF:
    def __init__(self, a: float = 0, c: float = 0):
        self.a = a
        self.c = c


def sigmf(x: ndarray, points: SigMF) -> ndarray:
    """
    :param x: 1d array or iterable values, independent variable
    :param points: [a, c] points of value for sigmoid
    :return: 1d array
    """
    a = points.a
    c = points.c
    values = []
    for n in range(len(x)):
        x0 = e ** (-a * (x[n] - c))
        x1 = 1 + x0
        values.append((1 + x1) ** -1)
    return array(values)

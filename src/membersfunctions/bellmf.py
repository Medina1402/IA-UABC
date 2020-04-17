from numpy import ndarray, array


# ==========================================================
# Member Functions ** Campana Generalizada **
# ==========================================================
class BellMF:
    def __init__(self, center: float = 0, width: float = 0, slope: float = 0):
        self.center = center
        self.width = width
        self.slope = slope


def bellmf(x: ndarray, points: BellMF) -> ndarray:
    """
    :param x: 1d array or iterable values, independent variable
    :param points: [a, b, c] points of value for bell generalized
    :return: 1d array
    """
    a = points.center
    b = points.width
    c = points.slope
    values = []
    for n in range(len(x)):
        try:
            x0 = ((x[n] - c) / a) ** 2
            x1 = 1 + (x0 ** b)
            values.append(x1 ** -1)
        except ZeroDivisionError:
            values.append(0)
    return array(values)

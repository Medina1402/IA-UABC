from numpy import ndarray, array
from . import getSlope


# ==========================================================
# Member Functions ** Trapesoidal **
# ==========================================================
class TrapMF:
    def __init__(self, leftBottom: float = 0, leftTop: float = 0, rightTop: float = 0, rightBottom: float = 0):
        self.leftBottom = leftBottom
        self.leftTop = leftTop
        self.rightTop = rightTop
        self.rightBottom = rightBottom


def trapmf(x: ndarray, points: TrapMF) -> ndarray:
    """
    :param x: 1d array or iterable values, independent variable
    :param points: [a, b, c, d] points of value for trapezoidal
    :return: 1d array
    """
    point_a = points.leftBottom
    point_b = points.leftTop
    point_c = points.rightTop
    point_d = points.rightBottom
    values = []
    for n in range(len(x)):
        txp1 = getSlope(point_a, point_a, point_b, x[n])
        txp2 = getSlope(point_c, x[n], point_d, point_d)
        minimum = min(txp1, txp2, 1)
        values.append(max(minimum, 0))
    return array(values)

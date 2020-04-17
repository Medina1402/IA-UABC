from numpy import ndarray, array
from . import getSlope


# ==========================================================
# Member Functions ** Triangular **
# ==========================================================
class TriMF:
    def __init__(self, left: float = 0, top: float = 0, right: float = 0):
        self.left = left
        self.top = top
        self.right = right


def trimf(x: ndarray, points: TriMF) -> ndarray:
    """
    :param x: 1d array or iterable values, independent variable
    :param points: [a, b, c] points of value for triangle
    :return: 1d array
    """
    point_a = points.left
    point_b = points.top
    point_c = points.right
    values = []
    for n in range(len(x)):
        txp1 = getSlope(point_a, point_a, point_b, x[n])
        txp2 = getSlope(point_b, x[n], point_c, point_c)
        minimum = min(txp1, txp2)
        values.append(max(minimum, 0))
    return array(values)

from numpy import ndarray, array, e


# ==========================================================
# Member Functions ** GAUSSIAN **
# ==========================================================
def gaussmf(x: ndarray, mean: float = 0, sigma: float = 0) -> ndarray:
    """
    :param x: 1d array or iterable values, independent variable
    :param mean: center value
    :param sigma: standard deviation
    :return: 1d array
    """
    values = []
    for n in range(len(x)):
        try:
            y = ((x[n] - mean) / sigma) ** 2
            values.append(e ** (-.5 * y))
        except ZeroDivisionError:
            values.append(0)
    return array(values)

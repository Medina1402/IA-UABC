from .bellmf import bellmf
from .gaussmf import gaussmf
from .gauss2mf import gauss2mf
from .sigmf import sigmf
from .dsigmf import dsigmf
from .psigmf import psigmf
from .trapmf import trapmf
from .trimf import trimf


def getSlope(x1, y1, x2, y2):
    try:
        slope = (y2 - y1) / (x2 - x1)
    except ZeroDivisionError:
        slope = 0
    return slope

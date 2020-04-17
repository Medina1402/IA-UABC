from numpy import ndarray
from src.membersfunctions import *


def genparamt1fls(data: ndarray, mf_n: int = 2, mf_type: str = "gaussmf"):
    _, in_n = data.shape

    for i in range(in_n):
        if mf_type == "gbellmf":
            pass
            # bellmf()
        elif mf_type == "gaussmf":
            pass
            # gaussmf()
        elif mf_type == "gauss2mf":
            pass
            # gauss2mf()
        elif mf_type == "dsigmf":
            pass
            # dsigmf()
        elif mf_type == "psigmf":
            pass
            # psigmf()
        elif mf_type == "trimf":
            pass
            # trimf()
        elif mf_type == "trapmf":
            trapmf()
        else:
            print(">> mf_type = %s\n>> Unsupported MF type!" % mf_type)

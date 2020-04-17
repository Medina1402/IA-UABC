from numpy import ndarray
from .sigmf import sigmf, Sig2MF, SigMF


# ==========================================================
# Member Functions ** Diferencia SIGMOIDAL **
# ==========================================================
def dsigmf(x: ndarray, params: Sig2MF):
    x0 = sigmf(x, SigMF(params.a1, params.c1))
    x1 = sigmf(x, SigMF(params.a2, params.c2))
    return x0 - x1

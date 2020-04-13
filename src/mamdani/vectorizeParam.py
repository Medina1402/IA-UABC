from numpy import ndarray, array
from src.mamdani import DesignParams, vectorizeMatrixTocolVec


def vectorizeParam(designParam: DesignParams) -> ndarray:
    SIGMA = vectorizeMatrixTocolVec(designParam.sigma)
    CENTER = vectorizeMatrixTocolVec(designParam.center)
    THETA = vectorizeMatrixTocolVec(designParam.theta)
    V = array([SIGMA[:], CENTER[:], THETA[:]])
    return V.ravel()

from numpy import array
from src.mamdani import mamcost1flscalcperf, mamcost1flscalcgx, mamcost1flscalcgrad
from src.mamdani.typedata import Tuple, DesignParams
from src.other import loadMatlabFile


def ejemplo1():
    engineInputs, engineTargets, testV, trainV, valV = loadMatlabFile("assets/engine_data.mat", "engineInputs",
                                                                      "engineTargets", "testV", "trainV", "valV")
    trainV = Tuple(array(trainV[0][0][2]), array(trainV[0][0][3]), trainV[0][0][4][0])

    CENTER, THETA, SIGMA = loadMatlabFile("assets/initdesignparam.mat", "CENTER", "CENTROID", "SIGMA")
    dsp = DesignParams(array(SIGMA), array(CENTER), array(THETA))
    SSE, Y, E, PHI, ALPHA = mamcost1flscalcperf(dsp, trainV.X.transpose(), trainV.T.transpose())
    print("SSE: ", SSE)

    MSE = SSE / Y.size
    print("MSE: ", MSE)

    RMSE = MSE ** .5
    print("RMSE: ", RMSE)

    gX, normgX, Jew, ew = mamcost1flscalcgx(dsp, trainV.X.transpose(), trainV.T.transpose(), Y, E, PHI)
    print("normgX", normgX)

    gX, normgradX = mamcost1flscalcgrad(dsp, trainV.X.transpose(), trainV.T.transpose(), Y, E, PHI)
    print("normgradX: ", normgradX)

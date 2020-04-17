from numpy import array, transpose, sqrt
from src.mamdani import *
from src.mamdani.typedata import DesignParams, Tuple, TrainParams
from src.other import loadMatlabFile

if __name__ == '__main__':
    engineInputs, engineTargets, testV, trainV, valV = loadMatlabFile("assets/engine_data.mat", "engineInputs",
                                                                      "engineTargets", "testV", "trainV", "valV")
    testV = Tuple(array(testV[0][0][2]), array(testV[0][0][3]), testV[0][0][4][0])
    trainV = Tuple(array(trainV[0][0][2]), array(trainV[0][0][3]), trainV[0][0][4][0])
    valV = Tuple(array(valV[0][0][2]), array(valV[0][0][3]), valV[0][0][4][0])

    CENTER, THETA, SIGMA = loadMatlabFile("assets/initdesignparam.mat", "CENTER", "CENTROID", "SIGMA")
    # x, tr = mamcost1flstraingdx(DesignParams(array(SIGMA), array(CENTER), array(THETA)), trainV, valV, testV, TrainParams(epochs=100))

    dsp = DesignParams(array(SIGMA), array(CENTER), array(THETA))
    SSE, Y, E, PHI, ALPHA = mamcost1flscalcperf(dsp, transpose(trainV.X), transpose(trainV.T))
    print("SSE: ", SSE)

    MSE = SSE / Y.size
    print("MSE: ", MSE)

    RMSE = sqrt(MSE)
    print("RMSE: ", RMSE)

    # CORREGUIR mamcost1flscalcgx
    gX, normgX, Jew = mamcost1flscalcgx(dsp, transpose(trainV.X), transpose(trainV.T), Y, E, PHI)
    print("normgX: ", normgX, " ")

    gX, normgradX = mamcost1flscalcgrad(dsp, transpose(trainV.X), transpose(trainV.T), Y, E, PHI)
    print("normgradX: ", normgradX)

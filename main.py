from numpy import array
from src.mamdani.mamcost1flstraingdx import mamcost1flstraingdx
from src.mamdani.typedata import DesignParams, Tuple, TrainParams
from src.other import loadMatlabFile

if __name__ == '__main__':
    engineInputs, engineTargets, testV, trainV, valV = loadMatlabFile("assets/engine_data.mat", "engineInputs",
                                                                      "engineTargets", "testV", "trainV", "valV")
    testV = Tuple(array(testV[0][0][2]), array(testV[0][0][3]), testV[0][0][4][0])
    trainV = Tuple(array(trainV[0][0][2]), array(trainV[0][0][3]), trainV[0][0][4][0])
    valV = Tuple(array(valV[0][0][2]), array(valV[0][0][3]), valV[0][0][4][0])

    CENTER, THETA, SIGMA = loadMatlabFile("assets/initdesignparam.mat", "CENTER", "CENTROID", "SIGMA")
    x, tr = mamcost1flstraingdx(DesignParams(SIGMA, CENTER, THETA), trainV, valV, testV, TrainParams(epochs=100))

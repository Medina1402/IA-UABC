from numpy import array, transpose
from src.mamdani import mamcost1flscalcgx, mamcost1flscalcgrad, mamcost1flscalcperf, mamcost1flstraingdx
from src.mamdani.typedata import DesignParams, Tuple, TrainParams
from src.other import loadMatlabFile


def ejemplo2():
    engineInputs, engineTargets, testV, trainV, valV = loadMatlabFile(
        "assets/engine_data.mat",
        "engineInputs",
        "engineTargets",
        "testV",
        "trainV",
        "valV"

    )
    testV = Tuple(array(testV[0][0][2]), array(testV[0][0][3]), array(testV[0][0][4][0]))
    trainV = Tuple(array(trainV[0][0][2]), array(trainV[0][0][3]), array(trainV[0][0][4][0]))
    valV = Tuple(array(valV[0][0][2]), array(valV[0][0][3]), array(valV[0][0][4][0]))
#
    CENTER, THETA, SIGMA = loadMatlabFile("assets/initdesignparam.mat", "CENTER", "CENTROID", "SIGMA")
    x, tr = mamcost1flstraingdx(
        DesignParams(array(SIGMA), array(CENTER), array(THETA)),
        trainV,
        valV,
        testV,
        TrainParams(epochs=20)
    )
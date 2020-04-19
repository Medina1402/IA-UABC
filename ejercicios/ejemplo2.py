from numpy import array
from src.mamdani.typedata import DesignParams, Tuple, TrainParams
from src.mamdani import mamcost1flstraingdx, mamcost1flscalcperf
from src.other import loadMatlabFile
from src.other.plotperf import plotregression


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
    engineInputs = array(engineInputs)
    engineTargets = array(engineTargets)
    SIGMA, CENTER, THETA = loadMatlabFile("assets/initdesignparam.mat", "SIGMA", "CENTER", "CENTROID")

    trainparams = TrainParams(
        epochs=1000,
        goal=1e-5,
        lr=0.01,
        lr_dec=0.7,
        lr_inc=1.05,
        max_fail=20,
        max_perf_inc=1.04,
        mc=0.9,
        min_grad=1e-6,
        show=5
    )
    designParam, tr = mamcost1flstraingdx(
        DesignParams(array(SIGMA), array(CENTER), array(THETA)),
        trainV,
        valV,
        testV,
        trainparams
    )

    SEE, Y, _, _, _ = mamcost1flscalcperf(designParam, engineInputs.transpose(), engineTargets.transpose())
    plotregression(engineTargets, Y.transpose())

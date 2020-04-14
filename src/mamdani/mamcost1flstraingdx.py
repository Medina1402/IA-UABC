from numpy import transpose, append, remainder, isfinite, add
from src.mamdani.mamcost1flscalcgrad import mamcost1flscalcgrad
from src.mamdani.mamcost1flscalcperf import mamcost1flscalcperf
from src.mamdani.reshapeParam import reshapeParam
from src.mamdani.typedata import Tuple, TrainParams, TR, VV, DesignParams
from src.mamdani.vectorizeParam import vectorizeParam
from src.matrix import vectorizeMatrixTocolVec
from src.other.plotperf import plotperf


def mamcost1flstraingdx(desingParam: DesignParams, train: Tuple, valV: Tuple = False, testV: Tuple = False,
                        trainsParam: TrainParams = TrainParams()) -> (DesignParams, TR):
    """
    :param desingParam:
    :param train:
    :param valV:
    :param testV:
    :param trainsParam:
    :return:
    """

    this = "MAMCOST1FLSTRAINGDX"
    performFcn = "SSE"
    stop = ""

    epochs = trainsParam.epochs
    tol = trainsParam.goal  # GOAL
    lr = trainsParam.lr
    lrinc = trainsParam.lr_inc
    lrDec = trainsParam.lr_dec
    maxFail = trainsParam.max_fail
    maxPerfInc = trainsParam.max_perf_inc
    mc = trainsParam.mc
    minGrad = trainsParam.min_grad
    show = trainsParam.show

    doVal = valV
    doTest = testV
    if doVal is not False:
        doVal = True
    if doTest is not False:
        doTest = True

    X = vectorizeParam(desingParam)
    perf, Y, E, PHI, ALPHA = mamcost1flscalcperf(desingParam, transpose(train.X), transpose(train.T))
    gX, normgX = mamcost1flscalcgrad(desingParam, transpose(train.X), transpose(train.T), Y=Y, E=E, PHI=PHI)
    dX = -1 * lr * gX

    vv = VV()
    if doVal:
        vv.designParam = desingParam
        vperfX, __a, __b, __c, __d = mamcost1flscalcperf(desingParam, transpose(valV.X), transpose(valV.T))
        vv.perf = vperfX
        vv.numfail = 0

    tr = TR()
    for epoch in range(epochs):
        tr.epoch = append(tr.epoch, epoch)
        tr.perf = append(tr.perf, perf)
        tr.lr = append(tr.lr, lr)

        if doVal:
            tr.vperf = append(tr.vperf, vv.perf)
        if doTest:
            temp, _Y, _E, _PHI, _ALPHA = mamcost1flscalcperf(desingParam, transpose(testV.X), transpose(testV.T))
            tr.tperf = append(tr.tperf, temp)

        if perf <= tol:
            stop = "Performance goal met."
        elif epoch == epochs:
            stop = "Maximum epoch reached, performance goal was not met."
        elif normgX < minGrad:
            stop = "Minimum gradient reached, performance goal was not met."
        elif doVal and (vv.numfail > maxFail):
            stop = "Validation stop."
            desingParam = vv.designParam

        # Progreso
        if not remainder(epochs, show) or len(stop):
            print("*** ", this, " ***")
            if isfinite(epochs):
                print(" -> Epoch:  %d / %d" % (epoch, epochs))
            if isfinite(tol):
                print(" -> %s:  %f / %f" % (performFcn.upper(), perf, tol))
            if isfinite(minGrad):
                print(" -> Gradient:  %f / %f" % (normgX, minGrad))
            plotperf(tr, tol, this, epoch)
            print("\n")
            if len(stop):
                print(" >>>> %s, %s" % (this, stop))

        dX = mc * dX - (1 - mc) * lr * gX
        X2 = X + dX
        desingParam2 = reshapeParam(desingParam, X2)
        perf2, Y2, E2, PHI2, ALPHA2 = mamcost1flscalcperf(desingParam2, X=transpose(train.X), T=transpose(train.T))
        if (perf2 / perf) > maxPerfInc:
            lr *= lrDec
            dX = lr * gX
        else:
            if perf2 < perf:
                lr = lr * lrinc
            X = X2
            desingParam = desingParam2
            perf = perf2
            gX, normgX = mamcost1flscalcgrad(desingParam2, transpose(train.X), transpose(train.T), Y2, E2, PHI2)

        if doVal:
            vperf, _a, _b, _c, _d = mamcost1flscalcperf(desingParam, transpose(valV.X), transpose(valV.T))
            if vperf < vv.perf:
                vv.perf = vperf
                vv.designParam = desingParam
                vv.numfail = 0
            elif vperf > vv.perf:
                vv.numfail += 1
    return desingParam, tr

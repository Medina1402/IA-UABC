from numpy import transpose, append, remainder, isfinite, ndarray

from src.mamdani.mamcost1flscalcgrad import mamcost1flscalcgrad
from src.mamdani.mamcost1flscalcperf import mamcost1flscalcperf
from src.mamdani.reshapeParam import reshapeParam
from src.mamdani.typedata import Tuple, TrainParams, TR, VV, DesignParams
from src.mamdani.vectorizeParam import vectorizeParam


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

    X: ndarray = vectorizeParam(desingParam)
    perf, Y, E, PHI, ALPHA = mamcost1flscalcperf(desingParam, train.X.transpose(), train.T.transpose())
    gX, normgX = mamcost1flscalcgrad(desingParam, train.X.transpose(), train.T.transpose(), Y, E, PHI)
    dX: ndarray = -1 * lr * gX

    vv = VV()
    if doVal:
        vv.designParam = desingParam
        vperfX, __a, __b, __c, __d = mamcost1flscalcperf(desingParam, valV.X.transpose(), valV.T.transpose())
        vv.perf = vperfX
        vv.numfail = 0

    tr = TR()
    for epoch in range(epochs):
        tr.epoch = append(tr.epoch, epoch)
        tr.perf = append(tr.perf, perf)
        tr.lr = append(tr.lr, lr)

        if doVal is True:
            tr.vperf = append(tr.vperf, vv.perf)
        if doTest is True:
            temp, _Y, _E, _PHI, _ALPHA = mamcost1flscalcperf(desingParam, testV.X.transpose(), testV.T.transpose())
            tr.tperf = append(tr.tperf, temp)

        stop = ""
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
        if not remainder(epochs, show) or len(stop) > 0:
            strTemp = this + " >>"
            if isfinite(epochs):
                strTemp = strTemp + " Epoch: " + str(epoch+1) + "/" + str(epochs)
            if isfinite(tol):
                strTemp = strTemp + ", " + performFcn.upper() + ": " + str(perf) + "/" + str(tol)
            if isfinite(minGrad):
                strTemp = strTemp + ", Gradient: " + str(normgX) + "/" + str(minGrad)
            # =============================
            # plotperf(tr, tol, this, epoch)
            # =============================
            print(strTemp)
            if len(stop) > 1:
                print(" >>>> %s, %s\n" % (this, stop))

        dX = (mc*dX)-(1-mc)*(lr*gX)
        X2 = X + dX
        desingParam2 = reshapeParam(desingParam, X2)
        perf2, Y2, E2, PHI2, ALPHA2 = mamcost1flscalcperf(desingParam2, train.X.transpose(), train.T.transpose())
        if (perf2 / perf) > maxPerfInc:
            lr *= lrDec
            dX = lr * gX
        else:
            if perf2 < perf:
                lr *= lrinc
            X = X2
            desingParam = desingParam2
            perf = perf2
            gX, normgX = mamcost1flscalcgrad(desingParam2, train.X.transpose(), train.T.transpose(), Y2, E2, PHI2)

        if doVal is True:
            vperf, _a, _b, _c, _d = mamcost1flscalcperf(desingParam, valV.X.transpose(), valV.T.transpose())
            if vperf < vv.perf:
                vv.perf = vperf
                vv.designParam = desingParam
                vv.numfail = 0
            elif vperf > vv.perf:
                vv.numfail += 1
    return desingParam, tr

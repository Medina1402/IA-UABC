from numpy import transpose, append, remainder, isfinite, ndarray, add

from src.mamdani.mamcost1flscalcgrad import mamcost1flscalcgrad
from src.mamdani.mamcost1flscalcperf import mamcost1flscalcperf
from src.mamdani.reshapeParam import reshapeParam
from src.mamdani.typedata import Tuple, TrainParams, TR, VV, DesignParams
from src.mamdani.vectorizeParam import vectorizeParam


def mamcost1flstraingdx(designParam: DesignParams, train: Tuple, valV: Tuple = False, testV: Tuple = False,
                        trainsParam: TrainParams = False) -> (DesignParams, TR):
    """
    :param trainsParam: TrainParams
    :param designParam:
    :param train:
    :param valV:
    :param testV:
    :param trainsParam:
    :return:
    """

    this = "MAMCOST1FLSTRAINGDX"
    performFcn = "SSE"

    if trainsParam is False:
        trainsParam = TrainParams()

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

    X: ndarray = vectorizeParam(designParam)
    perf, Y, E, PHI, ALPHA = mamcost1flscalcperf(designParam, train.X.transpose(), train.T.transpose())
    gX, normgX = mamcost1flscalcgrad(designParam, train.X.transpose(), train.T.transpose(), Y, E, PHI)
    dX: ndarray = -lr * gX

    vv = VV()
    tr = TR()
    if doVal:
        vv.designParam = designParam
        vperfX, _, _, _, _ = mamcost1flscalcperf(designParam, valV.X.transpose(), valV.T.transpose())
        vv.perf = vperfX
        vv.numfail = 0

    for epoch in range(epochs+1):
        tr.epoch = append(tr.epoch, epoch)
        tr.perf = append(tr.perf, perf)
        tr.lr = append(tr.lr, lr)

        if doVal:
            tr.vperf = append(tr.vperf, vv.perf)
        if doTest:
            temp, _, _, _, _ = mamcost1flscalcperf(designParam, testV.X.transpose(), testV.T.transpose())
            tr.tperf = append(tr.tperf, temp)

        stop = ""
        if perf <= tol:
            stop = "Performance goal met."
        elif epoch == epochs:
            stop = "Maximum epoch reached, performance goal was not met."
        elif normgX < minGrad:
            stop = "Minimum gradient reached, performance goal was not met."
        elif (doVal is True) and (vv.numfail > maxFail):
            stop = "Validation stop."
            designParam = vv.designParam

        # Progreso
        if ((epochs >= epoch) and (epoch % show == 0)) or len(stop):
            strTemp = this + " >>"
            if isfinite(epochs):
                strTemp = strTemp + " Epoch: " + str(epoch) + "/" + str(epochs)
            if isfinite(tol):
                strTemp = strTemp + ", " + performFcn.upper() + ": " + str(perf) + "/" + str(tol)
            if isfinite(minGrad):
                strTemp = strTemp + ", Gradient: " + str(normgX) + "/" + str(minGrad)
            # =============================
            # plotperf(tr, tol, this, epoch)
            # =============================
            print(strTemp)
            if len(stop):
                print(" >>>> %s, %s\n" % (this, stop))
                break

        dX = mc * dX - (1 - mc) * lr * gX
        X2 = X + dX
        designParam2 = reshapeParam(designParam, X2)
        perf2, Y2, E2, PHI2, _ = mamcost1flscalcperf(designParam2, train.X.transpose(), train.T.transpose())
        if (perf2/perf) > maxPerfInc:
            lr *= lrDec
            dX = lr * gX
        else:
            if perf2 < perf:
                lr *= lrinc
        X = X2
        designParam = designParam2
        perf = perf2
        gX, normgX = mamcost1flscalcgrad(designParam2, train.X.transpose(), train.T.transpose(), Y2, E2, PHI2)

    if doVal:
        vperf, _, _, _, _ = mamcost1flscalcperf(designParam, valV.X.transpose(), valV.T.transpose())
        if vperf < vv.perf:
            vv.perf = vperf
            vv.designParam = designParam
            vv.numfail = 0
        elif vperf > vv.perf:
            vv.numfail += 1
    return designParam, tr

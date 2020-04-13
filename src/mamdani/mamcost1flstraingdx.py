from numpy import transpose, append, remainder, isfinite, add
from src.mamdani import DesignParams, mamcost1flscalcperf, mamcost1flscalcgrad, vectorizeMatrixTocolVec
from src.mamdani.reshapeParam import reshapeParam
from src.mamdani.typedata import Tuple, TrainParams, TR, VV
from src.mamdani.vectorizeParam import vectorizeParam
from src.other import plotperf


def mamcost1flstraingdx(desingParam: DesignParams, train: Tuple, valV: Tuple = None, testV: Tuple = None, trainsParam: TrainParams = None) -> (DesignParams, TR):
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
    tol = trainsParam.tol  # GOAL
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
    if doVal is None:
        doVal = False
    if doVal is None:
        doTest = False

    stop = ""
    X = vectorizeParam(desingParam)
    perf, Y, E, PHI, ALPHA = mamcost1flscalcperf(desingParam, X=transpose(train.X), T=transpose(train.T))
    gX, normgX = mamcost1flscalcgrad(desingParam, transpose(train.X), transpose(train.T), Y, E, PHI)
    dX = -lr * gX

    vv = VV()
    if doVal is not None:
        vv.designParam = desingParam
        vv.perf = mamcost1flscalcperf(desingParam, transpose(valV.X), transpose(valV.T))
        vv.numfail = 0

    tr = TR()
    for epoch in range(epochs):
        tr.epoch = append(tr.epoch, epoch)
        tr.perf = append(tr.perf, perf)
        tr.lr = append(tr.lr, lr)

        if doVal:
            tr.vperf = append(tr.vperf, vv.perf)
        if doTest:
            temp = mamcost1flscalcperf(desingParam, transpose(testV.X), transpose(testV.T))
            tr.tperf = append(tr.tperf, temp)

        if perf <= tol:
            stop = "Performance goal met."
        elif epoch == epochs:
            stop = "Maximum epoch reached, performance goal was not met."
        elif normgX < minGrad:
            stop = "Minimum gradient reached, performance goal was not met."
        elif doVal is not None and vv.numfail > maxFail:
            stop = "Validation stop."
            desingParam = vv.designParam

        # Progreso
        if not remainder(epochs, show) or len(stop):
            print(this)
            if isfinite(epochs):
                print(" , Epoch %d/%d", epoch, epochs)
            if isfinite(tol):
                print(" , %s %d/%d", performFcn.upper(), perf, tol)
            if isfinite(minGrad):
                print(" , Gradient %d/%d", normgX, minGrad)
            print("\n")
            plotperf(tr, tol, this, epoch)
            if len(stop):
                print(" %s, %s", this, stop)

        dX = mc * dX - (1 - mc) * lr * gX
        X2 = add(X, dX)
        desingParam2 = reshapeParam(desingParam, X2)
        perf2, Y2, E2, PHI2, ALPHA2 = mamcost1flscalcperf(desingParam2, X=transpose(train.X), T=transpose(train.T))
        if perf2 / perf > maxPerfInc:
            lr = lr * lrDec
            dX = lr * gX
        else:
            if perf2 < perf:
                lr = lr * lrinc
            X = X2
            desingParam = desingParam2
            perf = perf2
            gX, normgX = mamcost1flscalcgrad(desingParam2, transpose(train.X), transpose(train.T), Y2, E2, PHI2)

        if doVal is not None:
            vperf = mamcost1flscalcperf(desingParam, transpose(valV.X), transpose(valV.T))
            if vperf < vv.perf:
                vv.perf = vperf
                vv.designParam = desingParam
                vv.numfail = 0
            elif vperf > vv.perf:
                vv.numfail += 1
    vecSigma = vectorizeMatrixTocolVec(desingParam.sigma)
    vecCenter = vectorizeMatrixTocolVec(desingParam.theta)
    vecTheta = vectorizeMatrixTocolVec(desingParam.theta)

    return DesignParams(vecSigma, vecCenter, vecTheta), tr

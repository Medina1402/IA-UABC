from numpy import ndarray, array


class DesignParams:
    def __init__(self, sigma: ndarray, center: ndarray, theta: ndarray):
        self.sigma = sigma
        self.center = center
        self.theta = theta


class Tuple:
    def __init__(self, X: ndarray, T: ndarray):
        self.X = X
        self.T = T


class VV:
    designParam: DesignParams
    perf: float
    numfail: float


class TR:
    def __init__(self, epoch: ndarray = array([]), perf: ndarray = array([]), lr:ndarray = array([]), vperf:ndarray = array([]), tperf:ndarray = array([]), mu: ndarray = array([])):
        self.epoch = epoch
        self.perf = perf
        self.vperf = vperf
        self.tperf = tperf
        self.mu = mu
        self.lr = lr


class TrainParams:
    def __init__(self, epochs=100, goal=0, lr=0.01, lr_inc=1.05, lr_dec=0.7, max_fail=5, max_perf_inc=1.04, mc=0.9,
                 min_grad=1e-6, show=25):
        self.epochs = epochs
        self.goal = goal
        self.lr = lr
        self.lr_inc = lr_inc
        self.lr_dec = lr_dec
        self.max_fail = max_fail
        self.max_perf_inc = max_perf_inc
        self.mc = mc
        self.min_grad = min_grad
        self.show = show

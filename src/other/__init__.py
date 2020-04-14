import numpy
from scipy.io import loadmat


# "assets/engine_data.mat"      -> engineInputs, engineTargets, testV, trainV, valV,
# "assets/initdesignparam.mat"  -> CENTER, CENTROID, SIGMA
# "assets/mg17_dataset.mat"     -> index, mgInputs, mgTargets, time, x_t
def loadMatlabFile(file, *args):
    data = []
    file = loadmat(file)
    for k in range(len(args)):
        data.append(numpy.array(file[args[k]]))
    return data

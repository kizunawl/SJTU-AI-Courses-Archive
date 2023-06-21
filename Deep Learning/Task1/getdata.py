import numpy as np
import jittor as jt

def getdata(trainpath, testpath):
    traindata = np.load(trainpath)
    testdata = np.load(testpath)

    traindata = jt.array(traindata)
    testdata = jt.array(testdata)

    return traindata, testdata
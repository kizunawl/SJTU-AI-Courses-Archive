import os
import numpy as np
import matplotlib as mpl
import torch

class PCAtest():
    def __init__(self, k):
        self.k = k

    def stand_data(self, data):
        mean_vector = np.mean(data, axis=0)
        return mean_vector, data-mean_vector

    def getCovMat(self, standData):
        return np.cov(standData, rowvar=0)
    
    def getFValueAndFVector(self, covMat):
        fValue, fVector = np.linalg.eig(covMat)
        return fValue, fVector

    def getVectorMatrix(self, fValue, fVector):
        fValueSort = np.argsort(-fValue)
        fValueTopN = fValueSort[:self.k]
        return fVector[:,fValueTopN]

    def getResult(self, data, vectorMat):
        return np.dot(data, vectorMat)

mpl.use('Agg')
import matplotlib.pyplot as plt

data = np.load(os.path.join('data', 'data4.npy'))
label = np.load(os.path.join('output', 'label4.npy'))

pca = PCAtest(2)
mean_vector, standdata = pca.stand_data(data)
cov_mat = pca.getCovMat(standdata)
fvalue, fvector = pca.getFValueAndFVector(cov_mat)
fvectormat = pca.getVectorMatrix(fvalue, fvector)
newdata = pca.getResult(standdata, fvectormat)
data = newdata

plt.scatter(data[:,0], data[:,1], c=label)
plt.axis('equal')
plt.savefig('data_example_.jpg')

#print([(data[i], label[i]) for i in range(10)])

import numpy as np

class PCA():
    def __init__(self):
        pass
    
    def execute(self, data: np.ndarray, k: int = 2):
        self.k = k
        mean_vec, std_data = self.stand_data(data)
        cov_mat = self.getCovMat(std_data)
        fval, fvec = self.getFValueAndFVector(cov_mat)
        fvec_mat = self.getVectorMatrix(fval, fvec)
        newdata = self.getResult(std_data, fvec_mat)
        return newdata

    def stand_data(self, data: np.ndarray):
        mean_vector = np.mean(data, axis=0)
        return mean_vector, data-mean_vector

    def getCovMat(self, standData: np.ndarray):
        return np.cov(standData, rowvar=0)
    
    def getFValueAndFVector(self, covMat: np.ndarray):
        fValue, fVector = np.linalg.eig(covMat)
        return fValue, fVector

    def getVectorMatrix(self, fValue: np.ndarray, fVector: np.ndarray):
        fValueSort = np.argsort(-fValue)
        fValueTopN = fValueSort[:self.k]
        return fVector[:,fValueTopN]

    def getResult(self, data: np.ndarray, vectorMat: np.ndarray):
        return np.dot(data, vectorMat)
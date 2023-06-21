import numpy as np
import random
import time


class SVM:
    def __init__(self, C: float = 1.0, 
                 kernel_weight: np.ndarray = [0, 1, 0], 
                 gamma: float = 1.0, c: float=1.0, 
                 d: float = 2) -> None:
        self.C = C
        self.gamma = gamma
        self.c = c
        self.d = d

        self.kernel_weight = kernel_weight
        self.kernel = self.kernel_multi

        # print(C, kernel_weight, gamma, c, d)

    def SMO(self, x: np.ndarray, y: np.ndarray, tol: int = 1e-3, maxIter: int = 40):
        m, n = x.shape
        b = 0.0
        alpha = np.zeros(m, dtype=np.float32)
        iter = 0
        while (iter < maxIter):
            count = 0
            for i in range(m):
                # E_i = np.sum(alpha * y * self.kernel(x, x[i])) + b - y[i]
                E_i = np.sum(alpha * y * self.K[i]) + b - y[i]
                if ((y[i] * E_i < -tol and alpha[i] < self.C) or (y[i] * E_i > tol and alpha[i] > 0)):
                    j = self.select(i, m)
                    if (y[i] != y[j]):
                        L = max(0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L = max(0, alpha[j] + alpha[i] - self.C)
                        H = min(self.C, alpha[j] + alpha[i])

                    # eta = self.kernel(x[i], x[i]) + self.kernel(x[j], x[j]) - 2 * self.kernel(x[i], x[j])
                    eta = self.K[i, i] + self.K[j, j] - 2 * self.K[i, j]
                    if (eta == 0):
                        continue

                    # E_j = np.sum(alpha * y * self.kernel(x, x[j])) + b - y[j]
                    E_j = np.sum(alpha * y * self.K[j]) + b - y[j]

                    new_alpha_j = alpha[j] + y[j] * (E_i - E_j) / eta
                    new_alpha_j = min(H, max(new_alpha_j, L))

                    new_alpha_i = alpha[i] + y[i] * y[j] * (alpha[j] - new_alpha_j)

                    # b1 = (b - E_i - y[i] * (new_alpha_i - alpha[i]) * self.kernel(x[i], x[i]) - y[j] * (new_alpha_j - alpha[j]) * self.kernel(x[i], x[j]))
                    b1 = (b - E_i - y[i] * (new_alpha_i - alpha[i]) * self.K[i, i] - y[j] * (new_alpha_j - alpha[j]) * self.K[i, j])
                    # b2 = (b - E_i - y[i] * (new_alpha_i - alpha[i]) * self.kernel(x[i], x[j]) - y[j] * (new_alpha_j - alpha[j]) * self.kernel(x[j], x[j]))
                    b2 = (b - E_i - y[i] * (new_alpha_i - alpha[i]) * self.K[i, j] - y[j] * (new_alpha_j - alpha[j]) * self.K[j, j])
                    if (0 < new_alpha_i and new_alpha_i < self.C):
                        b = b1
                    elif (0 < new_alpha_j and new_alpha_j < self.C):
                        b = b2
                    else:
                        b = (b1 + b2) / 2

                    alpha[i] = new_alpha_i
                    alpha[j] = new_alpha_j

                    count += 1

            if (count == 0):
                iter += 1
            else:
                iter = 0

        return alpha, b
    
    def kernel_Linear(self, x1: np.ndarray, x2: np.ndarray):
        return x1 @ x2
    
    def kernel_rbf(self, x1: np.ndarray, x2: np.ndarray):
        return np.exp(- self.gamma * ((np.linalg.norm((x1 - x2), axis=1)) ** 2))
    
    def kernel_poly(self, x1: np.ndarray, x2: np.ndarray):
        return (x1 @ x2 + self.c) ** self.d
    
    def kernel_multi(self, x1: np.ndarray, x2: np.ndarray):
        m = x1.shape[-1]
        x1 = np.reshape(x1, (-1, m))
        return self.kernel_weight @ np.array([self.kernel_Linear(x1, x2), 
                                              self.kernel_rbf(x1, x2),
                                              self.kernel_poly(x1, x2)])
    
    
    def fit(self, data: np.ndarray, label: np.ndarray):
        ti = time.time()
        m, n = data.shape
        self.X = data
        self.Y = label
        self.alpha = np.zeros(m, dtype=np.float32)
        self.b = 0.0

        self.K = np.zeros((m, m), dtype=np.float32)
        for i in range(m):
            self.K[i, :] = self.kernel(self.X, self.X[i])

        self.alpha, self.b = self.SMO(self.X, self.Y)


    def predict(self, x: np.ndarray):
        m, n = x.shape
        pred = np.zeros(m)
        for i in range(m):
            y = np.sum(self.alpha * self.Y * self.kernel(self.X, x[i])) + self.b
            pred[i] = 1 if y >= 0 else -1
        return pred

    def select(self, i: int, m: int):
        j = random.choice(range(m))
        while (j == i):
            j = random.choice(range(m))
        return j
    

class SVM_one2all(SVM):
    def __init__(self, C: float = 1.0, 
                 kernel_weight: np.ndarray = [0, 1, 0], 
                 gamma: float = 1.0, c: float = 1.0, 
                 d: float = 2.0) -> None:
        super().__init__(C, kernel_weight, gamma, c, d)
        
    def fit(self, data: np.ndarray, label: np.ndarray):
        m, n = data.shape
        class_size = np.unique(label).shape[0]
        self.X = data
        self.Y = np.zeros((class_size, m), dtype=np.int32)
        self.alpha = np.zeros((class_size, m), dtype=np.float32)
        self.b = np.zeros(class_size, dtype=np.float32)

        self.K = np.zeros((m, m), dtype=np.float32)
        for i in range(m):
            self.K[i, :] = self.kernel(self.X, self.X[i])

        for i in range(class_size):
            y = np.ones_like(label, dtype=np.int32)
            mask = (i != label)
            y[mask] = -1
            self.Y[i] = y
            self.alpha[i, :], self.b[i] = self.SMO(self.X, y)
    
    def predict(self, x: np.ndarray):
        m, n = x.shape
        pred = np.zeros(m)
        for i in range(m):
            y = np.sum(self.alpha * self.Y * self.kernel(self.X, x[i]), axis=1) + self.b
            pred[i] = np.argmax(y)
        return pred
    

class SVM_one2one(SVM):
    def __init__(self, C: float = 1.0, 
                 kernel_weight: np.ndarray = [0, 1, 0], 
                 gamma: float = 1.0, c: float = 1.0, 
                 d: float = 2.0) -> None:
        super().__init__(C, kernel_weight, gamma, c, d)

    def fit(self, data: np.ndarray, label: np.ndarray):
        m, n = data.shape
        self.class_size = np.unique(label).shape[0]
        classify_size = self.class_size * (self.class_size - 1) / 2
        self.X = []
        self.Y = []
        self.alpha = np.zeros((classify_size, m), dtype=np.float32)
        self.b = np.zeros(classify_size, dtype=np.float32)

        self.K = np.zeros((m, m), dtype=np.float32)
        for i in range(m):
            self.K[i, :] = self.kernel(self.X, self.X[i])

        for i in range(self.class_size):
            for j in range(i+1, self.class_size):
                count = i * self.class_size + j - i - 1

                mask_i = (i == label)
                mask_j = (j == label)
                size_i = label[mask_i].shape[0]
                size_j = label[mask_j].shape[0]

                x = np.zeros((size_i + size_j, n))
                y = np.zeros(size_i + size_j)
                x[:size_i] = data[mask_i]
                y[:size_i] = 1
                x[size_i:] = data[mask_j]
                y[size_i:] = -1
                self.X.append(x)
                self.Y.append(y)
                self.alpha[count, :], self.b[count] = self.SMO(x, y)
                # To use one2one, change the kernels in SMO live update

                count += 1

    def predict(self, x: np.ndarray):
        m, n = x.shape
        pred = np.zeros(m)
        for i in range(m):
            vote = np.zeros(self.class_size)
            for j in range(self.class_size):
                for k in range(j+1, self.class_size):
                    count = i * self.class_size + j - i - 1
                    y = np.sum(self.alpha[count] * self.Y[count] * self.kernel(self.X[count], x[i]), axis=1) + self.b[count]
                    vote[j if (y >= 0) else k] += 1
            pred[i] = np.argmax(vote)
        return pred

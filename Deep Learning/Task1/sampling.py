import numpy as np
import math
import matplotlib.pyplot as plt

PI = math.pi

def norm(x):
    return np.exp(- (x ** 2) / 2) / math.sqrt(2 * PI)


def x5(x):
    return 0.04 * x**5 + 0.4 * x**4 - 4.2 * x**3 + 0.2 * x**2 - x - 3


def expower(x):
    return 0.5 * np.exp(- 0.5 * x)


def makeSampling(func, funcName):
    sample_size = 1000
    train_size = 800
    test_size = 200
    x = np.random.uniform(-5, 5, sample_size)
    y = func(x)

    data = np.array((x, y))
    data = np.transpose(data)
    traindata = data[:train_size]
    testdata = data[train_size:]

    np.save(f'./dataset/{funcName}/train.npy', traindata)
    np.save(f'./dataset/{funcName}/test.npy', testdata)

    # plt.scatter(data[:,0], data[:,1])
    # plt.show()

def main():
    makeSampling(norm, 'norm')
    makeSampling(x5, 'x5')
    makeSampling(expower, 'exp')


if __name__=='__main__':
    main()
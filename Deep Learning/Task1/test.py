import model
import getdata
import jittor as jt
from jittor import nn
import numpy as np
import matplotlib.pyplot as plt
import os

def l2loss(pred, y):
    return (pred - y) ** 2


def test(trainPath, testPath, modelPath, func):
    if (jt.compiler.has_cuda):
        jt.flags.use_cuda = 1

    traindata, testdata = getdata.getdata(trainPath, testPath)

    net = model.Model()
    net.load_parameters(jt.load(modelPath))

    preds = []
    losses = []

    with jt.no_grad():
        for x, y in testdata:
            pred = net(x)
            preds.append(pred.item())
            loss = l2loss(pred, y)
            losses.append(loss.item())

    preds = np.array(preds)

    plt.cla()
    plt.scatter(testdata[:,0], testdata[:, 1], label='y')
    plt.scatter(testdata[:,0], preds, label='pred')
    plt.legend()
    plt.title(f'ave_loss = {round(np.average(losses), 4)}')
    plt.savefig(f'./figure/test_result_{func}.png')

if __name__ == '__main__':
    funcs = os.listdir('./dataset')
    for func in funcs:
        test(f'./dataset/{func}/train.npy', f'./dataset/{func}/test.npy', f'./model/{func}.pkl', func)
    # func = 'x5'
    # test(f'./dataset/{func}/train.npy', f'./dataset/{func}/test.npy', f'./model/{func}.pkl', func)
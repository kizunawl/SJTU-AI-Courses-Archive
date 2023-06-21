import model
import getdata
import jittor as jt
from jittor import nn
import numpy as np
import matplotlib.pyplot as plt
import os

def l2loss(pred, y):
    return 0.5 * (pred - y) ** 2

def train(trainPath, testPath, modelPath, func):
    if (jt.compiler.has_cuda):
        jt.flags.use_cuda = 1

    traindata, testdata = getdata.getdata(trainPath, testPath)

    net = model.Model()
    lr = 0.001
    optimizer = nn.SGD(net.parameters(), lr, momentum=0.9)

    loss_log = []
    for epoch in range(10):
        loss_per_epoch = []
        for x, y in traindata:
            pred = net(x)
            loss = l2loss(pred, y)
            # print(loss.item())
            loss_per_epoch.append(loss.item())
            optimizer.step(loss)
        loss_log.append(np.average(loss_per_epoch))

    net.save(modelPath)

    plt.cla()
    plt.plot(range(10), loss_log)
    plt.savefig(f'./figure/loss_{func}.png')

if __name__ == '__main__':
    funcs = os.listdir('./dataset')
    for func in funcs:
        train(f'./dataset/{func}/train.npy', f'./dataset/{func}/test.npy', f'./model/{func}.pkl', func)
    # func = 'norm'
    # train(f'./dataset/{func}/train.npy', f'./dataset/{func}/test.npy', f'./model/{func}.pkl', func)
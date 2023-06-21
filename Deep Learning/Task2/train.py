import numpy as np
import jittor as jt
import model
import matplotlib.pyplot as plt
import pickle
import dataloader
import parameter
import time

def train():
    param = parameter.parameters()
    if (jt.compiler.has_cuda):
        jt.flags.use_cuda = 1

    train_img, train_label = dataloader.load(train=True)

    print(len(train_label))

    # print(train_img[0])
    # print(train_label[0])

    net = model.AlexNet()

    optimizer = jt.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    lossfun = jt.nn.CrossEntropyLoss()

    loss_log = []

    for _ in range(param.epochs):
        ti = time.time()
        loss_per_epoch = []
        for __, (input, label) in enumerate(zip(train_img, train_label)):
            input = jt.array(input, dtype=jt.float32)
            label = jt.array(label, dtype=jt.float32)
            # print(label.shape)
            output = net(input)
            optimizer.zero_grad()
            # print(output.shape)
            # print(label.shape)
            loss = lossfun(output, label)
            optimizer.step(loss)
            loss_per_epoch.append(loss.data[0])

        print(f'Epoch [{_}]: {np.average(loss_per_epoch):.3f} Time: {time.time()-ti:.3f}')
        loss_log.append(np.average(loss_per_epoch))

    net.save(f'./model/{param.data_type}.pkl')

    plt.plot(range(param.epochs), loss_log)
    plt.savefig(f'./figures/{param.data_type}_trainLoss.png')


if __name__ == '__main__':
    train()
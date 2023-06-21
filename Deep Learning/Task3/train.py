import numpy as np
import jittor as jt
import model
import matplotlib.pyplot as plt
import dataloader
import parameter
import os
import time

def train(weight: np.ndarray = np.ones(10)):
    param = parameter.parameters()
    epochs = param.epochs

    if (jt.compiler.has_cuda):
        jt.flags.use_cuda = 1

    train_img, train_label = dataloader.load()

    print(param.data_type)
    print(len(train_label))
    # print('label:', train_label.shape)

    # print(train_img[0])
    # print(train_label[0])

    net = model.Net(param.input_size, param.output_size, param.hidden_layer, param.batchsize)

    optimizer = jt.optim.Adam(net.parameters(), lr=0.02, weight_decay=1e-4)
    # weight = np.array([6, 7, 7, 7, 5, 1, 1, 1, 1, 1])
    # lossfun = jt.nn.CrossEntropyLoss(weight=jt.array(weight))
    lossfun = jt.nn.CrossEntropyLoss()

    loss_log = []
    
    for _ in range(epochs):
        loss_per_epoch = []
        ti = time.time()
        for __, (input, label) in enumerate(zip(train_img, train_label)):
            input = jt.array(input, dtype=jt.float32)
            label = jt.Var(label).int64()
            net.clear_cells()
            output = net(input)

            optimizer.zero_grad()
            loss = lossfun(output, label)
            optimizer.step(loss)
            loss_per_epoch.append(loss.data[0])
        print(f'Epoch [{_}]: {np.average(loss_per_epoch):.4f} Time: {time.time()-ti:.3f}')
        loss_log.append(np.average(loss_per_epoch))

    model_save_dir = './model'
    if (not os.path.exists(model_save_dir)):
        os.mkdir(model_save_dir)
    net.save(f'{model_save_dir}/{param.data_type}.pkl')

    fig_save_dir = './figures'
    if (not os.path.exists(fig_save_dir)):
        os.mkdir(fig_save_dir)
    plt.plot(range(epochs), loss_log)
    plt.savefig(f'{fig_save_dir}/{param.data_type}_trainLoss.png')


if __name__ == '__main__':
    train()
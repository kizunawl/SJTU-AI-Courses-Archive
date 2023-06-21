import numpy as np
import jittor as jt
import model
import matplotlib.pyplot as plt
import dataloader
import parameter
import os
import time


# class Lossfun(jt.Module):
#     def __init__(self, *args, **kw) -> None:
#         super().__init__(*args, **kw)
#         self.CEL = jt.nn.CrossEntropyLoss()
    
#     def execute(self, output, label) -> None:
#         y = jt.argmax(label, dim=1)[0]
#         loss = jt.zeros(1, dtype=jt.float32)
#         for i in range(4):
#             loss += self.CEL(output[i], y[i])
#         loss /= 4.0
#         return loss

# def lossfun(output: jt.Var, label: jt.Var):
#     crossEntrophyLoss = jt.nn.CrossEntropyLoss()
#     y = jt.argmax(label, dim=1)[0]
#     print(y)
#     loss = jt.Var(0)
#     for i in range(4):
#         loss += crossEntrophyLoss(output[i], y[i])
#     loss /= 4.0
#     print(loss)
#     return loss

def train():
    param = parameter.parameters()
    epochs = param.epochs
    print(f'kernel {param.kernel_size}')
    print(f'pad {param.pad}')

    if (jt.compiler.has_cuda):
        jt.flags.use_cuda = 1

    train_img, train_label = dataloader.load()
    train_img = train_img[:500]
    train_label = train_label[:500]

    print(param.data_type)
    print(len(train_label))

    net = model.Net(sinkhorn_norm = True)

    optimizer = jt.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    # optimizer = jt.optim.SGD(params=net.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-5)
    lossfun = jt.nn.MSELoss(reduction='mean')
    # lossfun = Lossfun()
    loss_log = []

    if (not os.path.exists('./results')):
        os.mkdir('./results')

    for _ in range(epochs):
        loss_per_epoch = []
        ti = time.time()
        for __, (input, label) in enumerate(zip(train_img, train_label)):
            input = jt.array(input, dtype=jt.float32)
            label = jt.array(label, dtype=jt.uint32)
            output = net(input)
            optimizer.zero_grad()
            loss = lossfun(output, label)
            # print(jt.argmax(output.data[0], dim=1)[0])
            # print(jt.argmax(label.data[0], dim=1)[0])
            # print('-----------------------------------')
            optimizer.step(loss)
            loss_per_epoch.append(loss.data[0])

        print(f'Epoch [{_}]: {np.average(loss_per_epoch):.4f} Time: {(time.time()-ti):.2f}')
        
        loss_log.append(np.average(loss_per_epoch))

    model_save_dir = './model'
    if (not os.path.exists(model_save_dir)):
        os.mkdir(model_save_dir)
    net.save(f'{model_save_dir}/{param.data_type}_pad{param.pad}_k{param.kernel_size}.pkl')

    fig_save_dir = './figures'
    if (not os.path.exists(fig_save_dir)):
        os.mkdir(fig_save_dir)
    plt.plot(range(epochs), loss_log)
    plt.savefig(f'{fig_save_dir}/{param.data_type}_pad{param.pad}_k{param.kernel_size}_trainLoss.png')


if __name__ == '__main__':
    train()
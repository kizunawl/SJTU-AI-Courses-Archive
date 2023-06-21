import numpy as np
import jittor as jt
import model
import pickle
import matplotlib.pyplot as plt
import dataloader
import parameter

def plot_AccPlot(result):
    Accuracy = result[:,1] / result[:, 0]
    TotalAccuracy = np.mean(Accuracy)
    print(f'Total Accuracy: {TotalAccuracy:.3}')
    for idx in range(10):
        print(f'Acc on [label {idx}]: {Accuracy[idx]:.3}')

    fig = plt.figure()
    # ax = plt.gca()
    # ax.spines['top'].set_visible(False)
    a = fig.add_subplot(1, 1, 1)
    a.bar(range(10), Accuracy, width=0.5, color='orange', label='Accuracy on Labels')
    a.set_ylim(0, 1)
    a.set_ylabel('Accuracy')
    a.set_xlabel('label')
    b = a.twiny()
    b.plot(range(10), np.ones(10) * TotalAccuracy, color='red', label='Overall Accuracy')
    for i in range(10):
        plt.text(i-0.1, Accuracy[i]+0.02, f'{Accuracy[i]:.3}')
    plt.text(9, TotalAccuracy-0.05, f'{TotalAccuracy:.3}', color='red')
    plt.savefig('./figures/Accuracy_test_drop.png')
    plt.show()


def plot_ConfMat(result, param):
    def getcolor(val):
        return ('white' if (val>500) else 'black')

    result = np.int32(result)

    fontDict = dict(fontsize=10, color='black', family='Consolas', 
                    weight='normal')

    plt.rcParams['figure.figsize']=(6, 6)
    plt.matshow(result, cmap='Blues')
    # plt.figure(figsize=(8, 8))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            plt.text(x=j-0.2, y=i, s=result[i, j], 
                     fontdict=dict(fontsize=10, color=getcolor(result[i,j]), 
                                   family='Consolas', weight='normal'))
    
    # plt.savefig('./figures/ConfusionMat_Original.png')
    # plt.savefig('./figures/ConfusionMat_Drop.png')
    plt.savefig(f'./figures/ConfusionMat_{param.data_type}.png', dpi=200)
    plt.show()


def test(batch_size=10, drop=True, augment=True):
    param = parameter.parameters()
    if (jt.compiler.has_cuda):
        jt.flags.use_cuda = 1

    test_img, test_label = dataloader.load(train=False)

    net = model.AlexNet()
    # net.load('./model/Original.pkl' if (drop==False) else './model/Drop.pkl')
    net.load(f'./model/{param.data_type}.pkl')

    # result = np.zeros((10, 2))
    result = np.zeros((10, 10))

    with jt.no_grad():
        for _, (input, label) in enumerate(zip(test_img, test_label)):
            output = net(input)
            for idx in range(batch_size):
                pred = jt.argmax(output[idx], dim=1)

                # print(output[idx])
                # print(pred)
                # print(pred[0].data[0])

                #---------Acc_Plot----------------
                # result[label[idx], 0] += 1
                # if (pred[0].data[0] == label[idx]):
                #     result[label[idx], 1] += 1

                #---------Confusion Mat------------
                result[label[idx], pred[0].data[0]] += 1

    plot_ConfMat(result, param)



if __name__ == '__main__':
    test()
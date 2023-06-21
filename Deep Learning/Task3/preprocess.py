import os
import struct
import numpy as np
import pickle
from utils import augmentation, dropping, unpickle
import parameter
import random


def load_mnist_train():
    image_dir = os.path.join('../MNIST/raw/train-images-idx3-ubyte')
    label_dir = os.path.join('../MNIST/raw/train-labels-idx1-ubyte')
    with open(label_dir, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    with open(image_dir, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(len(labels), 28*28)
    return images, labels

def load_mnist_test():
    image_dir = os.path.join('../MNIST/raw/t10k-images-idx3-ubyte')
    label_dir = os.path.join('../MNIST/raw/t10k-labels-idx1-ubyte')
    with open(label_dir, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(image_dir, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 28*28)
    return images, labels


def load():
    param = parameter.parameters()

    with open(f'{param.dataset_dir}/traindata', 'rb') as f:
        data = pickle.load(f)
    train_img, train_label = data[0], data[1]

    with open(f'{param.dataset_dir}/testdata', 'rb') as f:
        data = pickle.load(f)
    test_img, test_label = data[0], data[1]

    batchsize = param.batchsize

    train_img = np.reshape(np.array(train_img), (-1, 28, 28))
    train_label = np.array(train_label)

    if ('Drop' in param.data_type or 'Augment' in param.data_type):
        train_img, train_label = dropping(train_img, train_label, batchsize)

    if ('Augment' in param.data_type):
        train_img, train_label = augmentation(train_img, train_label, batchsize)
    
    print(len(train_img))
    print(len(train_label))
    print(len(test_img))
    print(len(test_label))

    train_img = np.reshape(np.array(train_img), (-1, batchsize, 28, 28))
    train_label = np.reshape(np.array(train_label), (-1, batchsize))
    test_img = np.reshape(np.array(test_img), (-1, batchsize, 28, 28))
    test_label = np.reshape(np.array(test_label), (-1, batchsize))

    print(train_img.shape)
    print(train_label.shape)
    print(test_img.shape)
    print(test_label.shape)

    dataset_dir = f'./dataset/{param.data_type}'
    if (not os.path.exists(dataset_dir)):
        os.mkdir(dataset_dir)

    with open(f'{dataset_dir}/traindata', 'wb') as f:
        pickle.dump((train_img, train_label), f)
    with open(f'{dataset_dir}/testdata', 'wb') as f:
        pickle.dump((test_img, test_label), f)


def split_data():
    ###---------------------------- CIFAR-10 Begin ------------------------ ###
    img = []
    label = []
    for idx in range(1, 6):
        obj = unpickle(f'../CIFAR-10/cifar-10-batches-py/data_batch_{idx}')
        label.extend(obj[b'labels'])
        img.extend(obj[b'data'])

    totalsize = len(label)

    data = []
    for idx in range(totalsize):
        data.append([img[idx], label[idx]])

    random.shuffle(data)

    shf_img = []
    shf_label = []

    for idx in range(totalsize):
        shf_img.append(data[idx][0])
        shf_label.append(data[idx][1])

    shf_img = np.reshape(np.array(shf_img), (-1, 3, 32, 32))
    shf_label = np.array(shf_label)

    print(totalsize)
    print(len(shf_img))
    print(len(shf_label))

    train_size = round(0.8 * totalsize)

    train_img = shf_img[:train_size]
    train_label = shf_label[:train_size]
    test_img = shf_img[train_size:]
    test_label = shf_label[train_size:]
    ###---------------------------- CIFAR-10 End ------------------------ ###


    ###---------------------------- MNIST Begin ------------------------ ###
    # train_img, train_label = load_mnist_train()
    # test_img, test_label = load_mnist_test()
    ###---------------------------- MNIST End ------------------------ ###

    dataset_dir = './dataset'
    if (not os.path.exists(dataset_dir)):
        os.mkdir(dataset_dir)

    with open(f'{dataset_dir}/traindata', 'wb') as f:
        pickle.dump((train_img, train_label), f)
    with open(f'{dataset_dir}/testdata', 'wb') as f:
        pickle.dump((test_img, test_label), f)



if __name__ == '__main__':
    # split_data()
    load()
    # main()
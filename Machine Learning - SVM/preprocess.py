import os
import struct
import numpy as np
import pickle
import random
import Args

def unpickle(file):
    with open(file, 'rb') as fo:
        obj = pickle.load(fo, encoding='bytes')
    return obj

def load_mnist_train():
    image_dir = os.path.join('D:/Documents/Lectures/AI3607_DL/Hw/MNIST/raw/train-images-idx3-ubyte')
    label_dir = os.path.join('D:/Documents/Lectures/AI3607_DL/Hw/MNIST/raw/train-labels-idx1-ubyte')
    with open(label_dir, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    with open(image_dir, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(len(labels), 28*28)
    return images, labels

def load_mnist_test():
    image_dir = os.path.join('D:/Documents/Lectures/AI3607_DL/Hw/MNIST/raw/t10k-images-idx3-ubyte')
    label_dir = os.path.join('D:/Documents/Lectures/AI3607_DL/Hw/MNIST/raw/t10k-labels-idx1-ubyte')
    with open(label_dir, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(image_dir, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 28*28)
    return images, labels


def load():
    args = Args.Args()
    dataset_dir = f'../dataset/{args.dataset}'
    with open(f'{dataset_dir}/traindata[raw]', 'rb') as f:
        data = pickle.load(f)
    train_img, train_label = data[0], data[1]

    with open(f'{dataset_dir}/testdata[raw]', 'rb') as f:
        data = pickle.load(f)
    test_img, test_label = data[0], data[1]

    if ('MNIST' in args.dataset):
        shape = (-1, 28, 28)
    if ('CIFAR' in args.dataset):
        shape = (-1, 3, 32, 32)

    train_img = np.reshape(np.array(train_img), newshape=shape)
    train_label = np.array(train_label)
    test_img = np.reshape(np.array(test_img), newshape=shape)
    test_label = np.array(test_label)

    print(train_img.shape)
    print(train_label.shape)
    print(test_img.shape)
    print(test_label.shape)

    with open(f'{dataset_dir}/traindata', 'wb') as f:
        pickle.dump((train_img, train_label), f)
    with open(f'{dataset_dir}/testdata', 'wb') as f:
        pickle.dump((test_img, test_label), f)


def split_train_test():
    args = Args.Args()

    if ('CIFAR' in args.dataset):
    ###---------------------------- CIFAR-10 Begin ------------------------ ###
        img = []
        label = []
        for idx in range(1, 6):
            obj = unpickle(f'D:/Documents/Lectures/AI3607_DL/Hw/CIFAR-10/cifar-10-batches-py/data_batch_{idx}')
            label.extend(obj[b'labels'])
            img.extend(obj[b'data'])

        classified_data = []
        for i in range(10):
            classified_data.append([])

        for x, y in zip(img, label):
            classified_data[int(y)].append(np.reshape(np.array(x), newshape=(3, 32, 32)))

        for i in range(10):
            random.shuffle(classified_data[i])

        traindata_raw, testdata_raw = [], []
        for i in range(10):
            for j in range(500):
                traindata_raw.append((classified_data[i][j], i))
            for j in range(500, 600, 1):
                testdata_raw.append((classified_data[i][j], i))
        
        random.shuffle(traindata_raw)
        random.shuffle(testdata_raw)

        train_img = np.zeros((5000, 3, 32, 32))
        train_label = np.zeros(5000)
        for i in range(5000):
            train_img[i, :, :, :] = traindata_raw[i][0]
            train_label[i] = traindata_raw[i][1]
        test_img = np.zeros((1000, 3, 32, 32))
        test_label = np.zeros(1000)
        for i in range(1000):
            test_img[i, :, :, :] = testdata_raw[i][0]
            test_label[i] = testdata_raw[i][1]
    ###---------------------------- CIFAR-10 End ------------------------ ###

    if ('MNIST'in args.dataset):
    ###---------------------------- MNIST Begin ------------------------ ###
        train_img, train_label = load_mnist_train()
        test_img, test_label = load_mnist_test()
    ###---------------------------- MNIST End ------------------------ ###

    dataset_dir = f'../dataset/{args.dataset}'
    if (not os.path.exists(dataset_dir)):
        os.mkdir(dataset_dir)

    with open(f'{dataset_dir}/traindata[raw]', 'wb') as f:
        pickle.dump((train_img, train_label), f)
    with open(f'{dataset_dir}/testdata[raw]', 'wb') as f:
        pickle.dump((test_img, test_label), f)



if __name__ == '__main__':
    split_train_test()
    load()
import os
import struct
import numpy as np
import pickle
import random
from utils import augmentation, dropping, unpickle
import parameter
import utils


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


def segment(img: np.ndarray, data_type: str, slice: int):
    img = np.reshape(np.array(img), (3, slice*16, slice*16))
    perm_img = []

    if ('horizontal' in data_type):
        perm_img.append((img[:, :8, :], 0))
        perm_img.append((img[:, 8:16, :], 1))
        perm_img.append((img[:, 16:24, :], 2))
        perm_img.append((img[:, 24:, :], 3))
    elif ('vertical' in data_type):
        perm_img.append((img[:, :, :8], 0))
        perm_img.append((img[:, :, 8:16], 1))
        perm_img.append((img[:, :, 16:24], 2))
        perm_img.append((img[:, :, 24:], 3))
    elif ('grid' in data_type):
        for i in range(slice):
            for j in range(slice):
                perm_img.append((img[:, i*16:i*16+16, j*16:j*16+16], i*slice+j))

    random.shuffle(perm_img)

    img_list = []
    label = np.zeros(shape=(slice*slice, slice*slice))
    for i in range(slice*slice):
        label[i][perm_img[i][1]] = 1
        img_list.append(perm_img[i][0])

    return img_list, label


def load():
    param = parameter.parameters()
    slice = param.slice

    with open(f'{param.dataset_dir}/traindata', 'rb') as f:
        data = pickle.load(f)
    train_img, train_label = data[0], data[1]

    with open(f'{param.dataset_dir}/testdata', 'rb') as f:
        data = pickle.load(f)
    test_img, test_label = data[0], data[1]

    batchsize = param.batchsize
    
    print(len(train_img))
    print(len(test_img))

    perm_train_img = []
    perm_train_label = []
    for img in train_img:
        img_list, label_list = segment(utils.enlarge(img, slice), param.data_type, slice)
        perm_train_img.append(img_list)
        perm_train_label.append(label_list)
    
    perm_train_img = np.array(perm_train_img)
    perm_train_label = np.array(perm_train_label)
    print(perm_train_img.shape)
    print(perm_train_label.shape)
    
    perm_test_img = []
    perm_test_label = []
    for img in test_img:
        img_list, label_list = segment(utils.enlarge(img, slice), param.data_type, slice)
        perm_test_img.append(img_list)
        perm_test_label.append(label_list)
    
    perm_test_img = np.array(perm_test_img)
    perm_test_label = np.array(perm_test_label)
    print(perm_test_img.shape)
    print(perm_test_label.shape)

    if ('horizontal' in param.data_type):
        perm_train_img = np.reshape(np.array(perm_train_img, dtype=np.float32), (-1, batchsize, 4, 3, 8, 32))
        perm_test_img = np.reshape(np.array(perm_test_img, dtype=np.float32), (-1, batchsize, 4, 3, 8, 32))
    elif ('vertical' in param.data_type):
        perm_train_img = np.reshape(np.array(perm_train_img, dtype=np.float32), (-1, batchsize, 4, 3, 32, 8))
        perm_test_img = np.reshape(np.array(perm_test_img, dtype=np.float32), (-1, batchsize, 4, 3, 32, 8))
    elif ('grid' in param.data_type):
        perm_train_img = np.reshape(np.array(perm_train_img, dtype=np.float32), (-1, batchsize, slice*slice, 3, 16, 16))
        perm_test_img = np.reshape(np.array(perm_test_img, dtype=np.float32), (-1, batchsize, slice*slice, 3, 16, 16))
    
    perm_train_label = np.reshape(np.array(perm_train_label), (-1, batchsize, slice*slice, slice*slice))
    perm_test_label = np.reshape(np.array(perm_test_label), (-1, batchsize, slice*slice, slice*slice))
    # test_label = np.reshape(np.array(test_label), (-1, batchsize))

    print(perm_train_img.shape)
    print(perm_train_label.shape)
    print(perm_test_img.shape)
    print(perm_test_label.shape)

    # print(perm_train_img[0, 0, 0, :, :])

    # perm_test_img /= 255.0
    # perm_train_img /= 255.0

    dataset_dir = f'./dataset/{param.data_type}'
    if (not os.path.exists(dataset_dir)):
        os.mkdir(dataset_dir)

    with open(f'{dataset_dir}/traindata_{param.batchsize}_{slice}x{slice}', 'wb') as f:
        pickle.dump((perm_train_img, perm_train_label), f)
    with open(f'{dataset_dir}/testdata_{param.batchsize}_{slice}x{slice}', 'wb') as f:
        pickle.dump((perm_test_img, perm_test_label), f)


def split_train_test():
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
    # split_train_test()
    load()
    # main()
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import pickle

def show(img):
    plt.imshow(trans(img))
    plt.show()

def trans(img):
    img = img.reshape((3, 32, 32))
    x = np.zeros((32, 32, 3), dtype=np.float32)
    for i in range(3):
        x[:,:,i] = img[i,:,:]
    return x


def dropping(img, label, batchsize):
    data_classified = []
    for idx in range(10):
        data_classified.append([])

    # for i in range(len(label)):
    #     for j in range(batchsize):
    #         data_classified[label[i][j]].append(img[i][j])

    for i in range(len(label)):
        data_classified[label[i]].append(img[i])

    data = []
    for idx in range(5):
        for i in range(round(0.1 * len(data_classified[idx]))):
            data.append((data_classified[idx][i], idx))
    
    for idx in range(5, 10):
        for i in range(len(data_classified[idx])):
            data.append((data_classified[idx][i], idx))

    random.shuffle(data)
    
    dropped_img = []
    dropped_label = []

    for idx in range(len(data)):
        dropped_img.append(data[idx][0])
        dropped_label.append(data[idx][1])
        if ((idx+1)%10==0 and len(data)-idx<=10):
            break

    return dropped_img, dropped_label


def rev(img):
    newimg = np.zeros_like(img, dtype=np.float32)
    newimg[:,:,:] = img[:, :, ::-1]
    return newimg

def pad(img):
    newimg = np.zeros((img.shape[0], img.shape[1]+8, img.shape[2]+8), dtype=np.float32)
    newimg[:, 4:-4, 4:-4] = img[:, :, :]
    return [newimg[:, :-8, :-8], newimg[:, 8:, :-8], newimg[:, :-8, 8:], 
            newimg[:, 8:, 8:]]

def pix(img):
    newimg = np.zeros_like(img, dtype=np.float32)
    newimg[:,:,:] = img[:,:,:]
    for i in range(newimg.shape[1]):
        for j in range(newimg.shape[2]):
            flag = random.random()
            if (flag < 0.1):
                newimg[:, i, j] = 255
    return newimg

def cyc(img):
    c = img.shape[0]
    newimg = np.zeros((c*2, img.shape[1], img.shape[2]), dtype=np.float32)
    newimg[:c,:,:] = img[:,:,:]
    newimg[c:,:,:] = img[:,:,:]
    return [newimg[1:c+1, :, :], newimg[2:c+2, :, :], newimg[c-1::-1,:,:], 
            newimg[c:0:-1,:,:], newimg[c+1:1:-1, :, :]]

def addNoise(img):
    mean = 0
    sigma = 25
    gaussNoise = np.random.normal(mean, sigma, (3, 32, 32))

    noised_img = img + gaussNoise
    noised_img = np.clip(noised_img, a_min=0, a_max=255)
    return noised_img

def augmentation(train_img, train_label, batchsize):
    newdata = []

    for i in range(len(train_label)):
        img = train_img[i]
        label = train_label[i]
        if (label>=5):
            newdata.append([img, label])
        else:
            revimg = rev(img)
            piximg = pix(img)
            newdata.append([img, label])
            newdata.append([revimg, label])
            # newdata.append([piximg, label])

            padded_img_list = pad(img)
            for padded_img in padded_img_list:
                newdata.append([padded_img, label])

            padded_rev_img_list = pad(revimg)
            for padded_img in padded_rev_img_list:
                newdata.append([padded_img, label])

            # padded_pix_img_list = pad(piximg)
            # for padded_img in padded_pix_img_list:
            #     newdata.append([padded_img, label])

            cyc_img_list = cyc(img)
            for cyc_img in cyc_img_list:
                newdata.append([pix(cyc_img), label])

            noisedimg = addNoise(img)
            newdata.append([noisedimg, label])
            padded_noise_img_list = pad(noisedimg)
            for padded_img in padded_noise_img_list:
                newdata.append([padded_img, label])

            
    
    random.shuffle(newdata)

    new_img = []
    new_label = []
    for i in range(len(newdata)):
        new_img.append(newdata[i][0])
        new_label.append(newdata[i][1])
        if (len(newdata)-i<=10 and i%10==9):
            break
    
    return new_img, new_label


def unpickle(file):
    with open(file, 'rb') as fo:
        obj = pickle.load(fo, encoding='bytes')
    return obj
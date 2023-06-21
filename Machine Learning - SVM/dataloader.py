import pickle
import Args
import numpy as np

def load(train=True):
    args = Args.Args()
    dataset_dir = f'../dataset/{args.dataset}'

    if (train == True):
        with open(f'{dataset_dir}/traindata', 'rb') as f:
            data = pickle.load(f)
    else:
        with open(f'{dataset_dir}/testdata', 'rb') as f:
            data = pickle.load(f)

    return data[0], data[1]
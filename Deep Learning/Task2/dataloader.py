import pickle
import parameter

def load(train=True):
    param = parameter.parameters()
    dataset_dir = param.dataset_dir
    data_type = param.data_type

    if (train == True):
        with open(f'{dataset_dir}/{data_type}/traindata', 'rb') as f:
            data = pickle.load(f)
    else:
        with open(f'{dataset_dir}/{data_type}/testdata', 'rb') as f:
            data = pickle.load(f)

    return data[0], data[1]
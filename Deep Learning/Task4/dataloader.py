import pickle
import parameter

def load(train=True):
    param = parameter.parameters()
    dataset_dir = param.dataset_dir
    data_type = param.data_type
    batchsize = param.batchsize

    if (train == True):
        with open(f'{dataset_dir}/{data_type}/traindata_{param.batchsize}_{param.slice}x{param.slice}', 'rb') as f:
            data = pickle.load(f)
        return data[0], data[1]
    else:
        with open(f'{dataset_dir}/{data_type}/testdata_{param.batchsize}_{param.slice}x{param.slice}', 'rb') as f:
            data = pickle.load(f)
        return data[0], data[1]
import numpy as np

def open_data(direc, ratio_train=0.8, dataset="ECG5000"):
    """Input:
    direc: location of the UCR archive
    ratio_train: ratio to split training and testset
    dataset: name of the dataset in the UCR archive"""
    datadir = direc + '/' + dataset + '/' + dataset
    data_train = np.loadtxt(datadir + '_TRAIN', delimiter=',')
    if(ratio_train < 1):
        data_test_val = np.loadtxt(datadir + '_TEST', delimiter=',')[:-1]
        data = np.concatenate((data_train, data_test_val), axis=0)
        data = np.expand_dims(data, -1)
    else:
        data = np.expand_dims(data_train, -1)

    N, D, _ = data.shape

    ind_cut = int(ratio_train * N)
    #ind = np.random.permutation(N)
    ind = range(N)
    return data[ind[:ind_cut], 1:, :], data[ind[ind_cut:], 1:, :], data[ind[:ind_cut], 0, :], data[ind[ind_cut:], 0, :]


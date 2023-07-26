'''
Describe: Prepare training data for models.
Author: Yi Hu
Email: yhu28@ncsu.edu
'''
import pandas as pd
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
# from FC_GAN import Discriminator, Generator
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pickle
from sys import platform

def load_data():
    if platform == "linux" or platform == "linux2":
        real_file = '/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/dataset/newRiver/GANData0.csv'
    elif platform == "darwin":
        real_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\GANData0.csv'
    elif platform == "win32":
        real_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\GANData0.csv'

    real_data = pd.read_csv(real_file)

    real_loads = real_data.iloc[:, 1:-1].to_numpy()

    return real_loads


# ===========================
# encode one batch
# batch_size = N
# N * 672 * 9 to N * 4 * 672 * 8
# ===========================
def encode_users(data, npoint=96*7, nuser=8, t1=2.0, t2=4.0, t3=6.0):
    print('=====welcome to encoding!!!====')
    data3 = torch.zeros([data.shape[0], 4, npoint, nuser], dtype=torch.float32)
    total = data3.shape[0]
    for i in range(data3.shape[0]):
        # print('Encoding %d/%d' % (i, total))
        for j in range(data3.shape[2]):
            # temperature   0-120
            temperature = data[i][j][-1] / 120.0
            for k in range(data3.shape[3]):
                load = data[i][j][k]
                if load < t1:  # green - blue
                    d1 = 0.0
                    d2 = 1.0 - load / t1
                    d3 = load / t1
                elif load < t2:  # bule - red
                    d1 = (load - t1) / (t2 - t1)
                    d2 = 0.0
                    d3 = 1.0 - (load - t1) / (t2 - t1)
                elif load < t3:  # red - black
                    d1 = 1.0 - (load - t2) / (t3 - t2)
                    d2 = 0.0
                    d3 = 0.0
                else:  # black
                    d1 = 0.0
                    d2 = 0.0
                    d3 = 0.0
                # print(load, d1, d2, d3)
                data3[i, 0, j, k] = d1
                data3[i, 1, j, k] = d2
                data3[i, 2, j, k] = d3
                data3[i, 3, j, k] = temperature

    # # transform from [0, 1] to [-1, 1]
    # data3 = data3 * 2.0 - 1.0
    return data3

# ===========================
# decode one batch
# batch_size = N
# N * 4 * 672 * 8 to N * 672 * 8
# ===========================
def decode_16users(imgs, npoint=672, nuser=8, t1=2.0, t2=4.0, t3=6.0):
    print("decoding...")
    loads = torch.zeros([imgs.shape[0], npoint, nuser], dtype=torch.float32)
    temperature = torch.zeros([imgs.shape[0], npoint, nuser], dtype=torch.float32)

    total = imgs.shape[0]
    for n in range(imgs.shape[0]):
        # print('Decoding %d/%d' % (n, total))
        for i in range(npoint):
            for j in range(nuser):
                r = imgs[n, 0, i, j]
                g = imgs[n, 1, i, j]
                b = imgs[n, 2, i, j]
                t = imgs[n, 3, i, j]
                load = 0
                if g < 0.05 and b < 0.05: # 4-6
                    load = (1 - r) * (t3 - t2) + t2
                else:
                    s = r + g + b
                    r = r / s
                    g = g / s
                    b = b / s
                    if r >= g:    # 2-4
                        load = r * (t2 - t1) + t1
                    else:    # 0-2
                        load = b * t1

                loads[n, i, j] = load
                temperature[n, i, j] = t * 120.0

    return loads, temperature

# ========================================
# evaluate loads
# N * 96 * 16
# ========================================
def evaluate_loads(loads):
    # print(loads)
    max_array = []
    mean_array = []
    npeak_array = []
    lf_array = []
    for n in range(loads.shape[0]):
        # figure, axis = plt.subplots(8, 2)
        for i in range(loads.shape[2]):
            max = torch.max(loads[n, :, i])
            mean = torch.mean(loads[n, :, i])
            peaks = find_peaks(loads[n, :, i], prominence=1, distance=8)
            # axis[i % 8, i // 8].plot(loads[n, :, i])
            # [axis[i % 8, i // 8].axvline(p, c='C3', linewidth=0.3) for p in peaks[0]]
            npeak = len(peaks[0])
            max_array.append(max.item())
            mean_array.append(mean.item())
            npeak_array.append(npeak)
            lf_array.append(mean.item()/max.item())
        # plt.plot(loads[n])
        # Adding legend, which helps us recognize the curve according to it's color
        # plt.legend()
        # plt.show()
    # print(max_array)
    # print(mean_array)
    plt.subplot(2, 2, 1)
    plt.hist(max_array)
    plt.title('max')
    plt.subplot(2, 2, 2)
    plt.hist(mean_array)
    plt.title('mean')
    plt.subplot(2, 2, 3)
    plt.hist(lf_array)
    plt.title('lf')
    plt.subplot(2, 2, 4)
    plt.hist(npeak_array)
    plt.title('npeak')
    plt.show()


def display():
    # Hyperparameters etc.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    # device = "cpu"
    lr = 5e-4
    z_dim = 128
    nuser = 100 + 2
    npoint = 96
    image_dim = npoint * nuser * 1  # 784
    batch_size = 9
    num_epochs = 101

    dataset = load_data()
    n_sample = dataset.shape[0]
    res = n_sample % (npoint*7)
    X = dataset[:n_sample - res, :]
    # print(n_sample, res, X.shape[0])


    X = X.reshape((-1, npoint, nuser))

    loader = DataLoader(X, batch_size=batch_size, shuffle=True)

    step = 0

    for epoch in range(num_epochs):
        for batch_idx, real in enumerate(loader):
            real = real.view(-1, npoint * nuser).to(device)

            print(len(real))
            if batch_idx == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} ")

                with torch.no_grad():
                    data = real.reshape(-1, npoint, nuser)
                    # data = real.reshape(-1, npoint, nuser)
                    data_former = encode_data_3(data, npoint, nuser)
                    data_now = encode_data(data, npoint, nuser)

                    import seaborn as sns
                    # plt.figure(figsize=(10, 10))
                    plt.subplot(1,2,2)
                    plt.imshow(data_former[0])
                    # plt.pcolormesh(data[0])
                    # plt.colorbar()
                    plt.xlabel('Users')
                    plt.ylabel('Load Profile')
                    plt.subplot(1,2,1)
                    plt.imshow(data_now[0, 0:3].permute(1, 2, 0))

                    plt.show()

                    step += 1

# display()
from torch.utils.data import Dataset


class NEWRIVERDataset(Dataset):
    def __init__(self, x, y):
        self.data = torch.from_numpy(x).float()
        self.label = torch.LongTensor(y)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


class newRiverLoader():

    def __init__(self):
        self.smid = []

    # not used
    def load_data_for_classifier(self):
        # file_name = 'C:\\Users\\yhu28\\Downloads\\BadSMsFixedv2\\SMs_3001_4000\\sm_74715523.pkl'
        # with open(file_name, 'rb') as f:
        #     data = pickle.load(f, protocol=4)
        #     print(data)
        trainDataFile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\classiferTrain0524.csv'
        testDataFile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\classiferTest0524.csv'
        trainData = pd.read_csv(trainDataFile)
        testData = pd.read_csv(testDataFile)

        return trainData, testData

    def load_data_for_groupGAN(self, day=0):
        print('Loading data for group GAN...')
        if platform == "linux" or platform == "linux2":
            trainDataFile = '~/Documents/Code/Research/MultiLoad-GAN_public/dataset/newRiver/GANData0.csv'
        elif platform == "darwin":
            trainDataFile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\GANData0.csv'
        elif platform == "win32":
            trainDataFile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\GANData0.csv'
        trainData_ori = pd.read_csv(trainDataFile)

        week_list = []
        nweek = int(len(trainData_ori.index)/672)
        for i in range(nweek):
            df_week = trainData_ori.iloc[i*672:(i+1)*672, 1:-1].copy()
            df_week = df_week.loc[:, df_week.mean().sort_values(ascending=True).index]
            df_week.columns = range(8)
            week_list.append(df_week)
        sortedTrainData = pd.concat(week_list)
        sortedTrainData['temp'] = trainData_ori.iloc[:, -1]

        trainData = sortedTrainData.to_numpy(dtype=float)
        trainData = trainData.reshape((-1, 96 * 7, 9))

        print("Encoding training data...")
        # torch.tensor
        trainData = encode_users(trainData, npoint=672, nuser=8)
        trainData = trainData * 2.0 - 1.0
        # print('Encoding data...')
        # X = encode_users(trainData, npoint=96*7, nuser=8)
        return trainData

        # print('Creating dataloader...')
        # train_loader = DataLoader(trainData, batch_size=batch_size, shuffle=True, num_workers=2)
        # print('Done!')
        # return train_loader

    def load_data_for_groupGAN_no_encoding(self, day=0):
        print('Loading data for group GAN...')
        if platform == "linux" or platform == "linux2":
            trainDataFile = '~/Documents/Code/Research/MultiLoad-GAN_public/dataset/newRiver/GANData0.csv'
        elif platform == "darwin":
            trainDataFile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\GANData0.csv'
        elif platform == "win32":
            trainDataFile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\GANData0.csv'
        trainData_ori = pd.read_csv(trainDataFile)

        week_list = []
        nweek = int(len(trainData_ori.index)/672)
        for i in range(nweek):
            df_week = trainData_ori.iloc[i*672:(i+1)*672, 1:-1].copy()
            df_week = df_week.loc[:, df_week.mean().sort_values(ascending=True).index]
            df_week.columns = range(8)
            week_list.append(df_week)
        sortedTrainData = pd.concat(week_list)

        trainData = sortedTrainData.to_numpy(dtype=float)
        trainData = trainData.reshape((-1, 96 * 7, 8))

        print("Encoding training data...")
        # torch.tensor
        trainData_torch = torch.zeros([trainData.shape[0], 1, 96*7, 8], dtype=torch.float32)
        for i in range(trainData.shape[0]):
            for j in range(96*7):
                for k in range(8):
                    value = trainData[i, j, k]
                    if value > 6.0:
                        trainData_torch[i, 0, j, k] = 1.0
                    else:
                        trainData_torch[i, 0, j, k] = value / 6.0

        trainData_torch = trainData_torch * 2.0 - 1.0
        return trainData_torch

    def load_weekly_shifted_data_for_classifier(self, starting_day=7, shuffle='origin'):
        # file_name = 'C:\\Users\\yhu28\\Downloads\\BadSMsFixedv2\\SMs_3001_4000\\sm_74715523.pkl'
        # with open(file_name, 'rb') as f:
        #     data = pickle.load(f, protocol=4)
        #     print(data)
        dfs_train = []
        num = min(starting_day+1, 7)
        for i in range(num):
            if shuffle == 'random':
                trainDataFile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\classifer'+str(i)+'.csv'
            elif shuffle == 'sort':
                if platform == "linux" or platform == "linux2":
                    trainDataFile = '~/Documents/Code/Research/MultiLoad-GAN_public/dataset/newRiver/classiferSorted' + str(i) + '.csv'
                elif platform == "darwin":
                    trainDataFile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\classiferSorted' + str(i) + '.csv'
                elif platform == "win32":
                    trainDataFile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\classiferSorted' + str(i) + '.csv'
            elif shuffle == 'origin':
                trainDataFile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\classiferOrigin' + str(i) + '.csv'
            else:
                print('!!! unknown shuffle !!!')
            # print("Read ", trainDataFile)
            trainData = pd.read_csv(trainDataFile)
            dfs_train.append(trainData)

        # traing set
        if starting_day == 7:
            trainset = pd.concat(dfs_train)
            return trainset
        else:  # 0-6
            return dfs_train[starting_day]


#     save shuffled data file
#     read Dataframe write numpy
#     randomly separate training set and testing set
#     for classifier
    def save_train_test_in_numpy(self, day=7, shuffle='origin'):
        XY = self.load_weekly_shifted_data_for_classifier(starting_day=day, shuffle=shuffle)
        XY = XY.iloc[:, 1:].to_numpy(dtype=float)
        XY = XY.reshape((-1, 96 * 7, 9))
        rand_indx = torch.randperm(XY.shape[0])
        XY = XY[rand_indx]
        X = XY[:, :, :-1]
        X = X.reshape((-1, 1, 96 * 7, 8))
        Y = XY[:, 0, -1]

        batch_size = 64
        percent = int(X.shape[0] * (1 - 0.2))
        train_x, train_y, test_x, test_y = X[:percent], Y[:percent], X[percent:], Y[percent:]
        train_set = NEWRIVERDataset(train_x, train_y)
        test_set = NEWRIVERDataset(test_x, test_y)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)
        if platform == "linux" or platform == "linux2":
            trainDataFile = '~/Documents/Code/Research/MultiLoad-GAN_public/dataset/newRiver/loaderTrain' + str(
                day) + '.pt'
            testDataFile = '~/Documents/Code/Research/MultiLoad-GAN_public/dataset/newRiver/loaderTest' + str(
                day) + '.pt'
        elif platform == "darwin":
            trainDataFile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\loaderTrain' + str(
                day) + '.pt'
            testDataFile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\loaderTest' + str(
                day) + '.pt'
        elif platform == "win32":
            trainDataFile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\loaderTrain' + str(
            day) + '.pt'
            testDataFile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\loaderTest' + str(
            day) + '.pt'
        torch.save(train_loader, trainDataFile)
        torch.save(test_loader, testDataFile)

    # single GAN
    def load_data_for_GAN(self):
        if platform == "linux" or platform == "linux2":
            file = '~/Documents/Code/Research/MultiLoad-GAN_public/dataset/newRiver/GANData0.csv'
        elif platform == "darwin":
            file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\GANData0.csv'
        elif platform == "win32":
            file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\GANData0.csv'
        data = pd.read_csv(file)    # already sorted

        # file_path = 'C:\\Users\\yhu28\\Documents\\Code\\Data\\new_river\\'
        # date_file = file_path + 'temp_Boone.csv'
        # d = pd.read_csv(date_file)
        # d = d.loc['2017-7-24':'2020-12-27']
        # l = len(d.index)
        # for i in range(len(data)//l):
        #     data['Temp'].iloc[i*l:i*l+l] = d['Temp'].astype(np.float32)  # 0-120

        return data

if __name__ == "__main__":
    loader = newRiverLoader()

    # save dataset in torch.utils.data DataLoader format
    # for i in range(1):
    #     loader.save_train_test_in_numpy(day=i, shuffle='sort')

    loader.save_train_test_in_numpy(day=0, shuffle='sort')
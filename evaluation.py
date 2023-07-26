'''
Describe: Evaluate the generated load profiles, and plot the results
Author: Yi Hu
Email: yhu28@ncsu.edu
'''
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
# import torchvision.transforms as transforms
# from torch import optim
import matplotlib.pyplot as plt
import seaborn as sns
import groupGAN
import Data
import SingleLoadGAN.singleUser_GAN as singleUser_GAN
import numpy as np
from sys import platform
from torch.utils.data import Dataset, DataLoader

from collections import OrderedDict
from ignite.metrics import FID
from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *

bhist = False
xlabel_distance = 1
ylabel_distance = 4
title_distance = 1.0

fig_width = 6
fig_height = 4
bottom_margin = 0.11
top_margin = 0.95
left_margin = 0.12
right_margin =0.95

def calculate_fid(real, fake):
    # ===============Frechet Inception distance =============================
    # create default evaluator for doctests
    def eval_step(engine, batch):
        return batch

    default_evaluator = Engine(eval_step)

    # create default optimizer for doctests
    param_tensor = torch.zeros([1], requires_grad=True)
    default_optimizer = torch.optim.SGD([param_tensor], lr=0.1)
    # create default trainer for doctests
    # as handlers could be attached to the trainer,
    # each test must define his own trainer using `.. testsetup:`
    def get_default_trainer():
        def train_step(engine, batch):
            return batch

        return Engine(train_step)
    # create default model for doctests
    default_model = nn.Sequential(OrderedDict([
        # ('base', nn.Linear(4, 2)),
        ('fc', nn.Linear(1, 1))
    ]))
    manual_seed(666)

    metric = FID(num_features=1, feature_extractor=default_model)
    metric.attach(default_evaluator, "fid")
    state = default_evaluator.run([[torch.from_numpy(real).reshape((-1, 1)).float(), torch.from_numpy(fake).reshape((-1, 1)).float()]])
    print("FID: ", state.metrics["fid"])
    return state.metrics["fid"]

class LSTMDataset(Dataset):
    def __init__(self, x):
        self.data = torch.from_numpy(x[:, :672]).float()
        self.label = torch.from_numpy(x[:, 672:]).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

class Evaluator:
    if platform == "linux" or platform == "linux2":
        workspace_dir = '/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/'
        ckpt_dir_single = os.path.join(workspace_dir, 'singleUser/checkpoints')
        ckpt_dir_lstm = os.path.join(workspace_dir, 'lstm/checkpoints')
    elif platform == "darwin":
        workspace_dir = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public'
        ckpt_dir_single = os.path.join(workspace_dir, 'singleUser\\checkpoints')
        ckpt_dir_lstm = os.path.join(workspace_dir, 'lstm/checkpoints')
    elif platform == "win32":
        workspace_dir = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public'
        ckpt_dir_single = os.path.join(workspace_dir, 'singleUser\\checkpoints')
        ckpt_dir_lstm = os.path.join(workspace_dir, 'lstm/checkpoints')
    ckpt_dir = os.path.join(workspace_dir, 'checkpoints')

    z_dim = 100
    n_output = 16
    npoint = 96*7
    nuser = 8

    def __init__(self):
        self.G = groupGAN.Generator(self.z_dim, dim=80)
        # self.G.load_state_dict(torch.load(os.path.join(self.ckpt_dir, 'G_Auto.pth')))
        self.G.load_state_dict(torch.load(os.path.join(self.ckpt_dir, 'G_Impressive.pth')))
        self.G.eval()
        self.G.cuda()

        self.singleUser_G = singleUser_GAN.Generator(self.z_dim)
        self.singleUser_G.load_state_dict(torch.load(os.path.join(self.ckpt_dir_single, 'G_single.pth')))
        self.singleUser_G.eval()
        self.singleUser_G.cuda()

    def savesingledata(self):
        times = 20
        # Generate 100 * 10 images and make a grid to save them.
        ysingle = []
        for i in range(times):
            # single generation
            z_sample_single = Variable(torch.randn(self.n_output * self.nuser, self.z_dim)).cuda()
            loads_single = self.singleUser_G(z_sample_single).data
            # transform from [-1, 1] to [0, 10]
            loads_single = loads_single * 5.0 + 5.0

            rand_indx = torch.randperm(len(loads_single))
            loads_single = loads_single[rand_indx, :, :]
            # print(loads_single)
            # yloads = torch.zeros([self.n_output, self.npoint, self.nuser], dtype=torch.float32)
            for i in range(self.n_output):
                groupdata = []
                for c in range(self.nuser):
                    n = i * self.nuser + c
                    load = loads_single[n, 0].reshape((self.npoint, 1))
                    groupdata.append(load)
                group = torch.cat(groupdata, dim=1)  # 1 day 8 users    96*7*8
                part = pd.DataFrame(group).astype(np.float32)
                part = part.loc[:, part.mean().sort_values(ascending=True).index]
                # group1 = torch.tensor(part.values)
                ysingle.append(part)
            # ysingle.append(yloads)
            print('Dis single generated loads')
            # self.dispLoads(yloads[0])
        Y_single = pd.concat(ysingle)
        print("data generated...")
        if platform == "linux" or platform == "linux2":
            datafile = '~/Documents/Code/Research/MultiLoad-GAN_public/dataset/newRiver/singleGeneratedData.csv'
        elif platform == "darwin":
            datafile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\singleGeneratedData.csv'
        elif platform == "win32":
            datafile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\singleGeneratedData.csv'
        Y_single.to_csv(datafile)

    def savelstmdata(self):
        newRiverloader = Data.newRiverLoader()
        X = newRiverloader.load_data_for_GAN()
        X = X.iloc[:, 1:9].to_numpy()
        X = X.reshape((-1), order='F')
        # X = singleUser_Data.load_data()  # numpy  columns=1
        X = X.reshape((-1, 672 * 2))
        print(np.amax(X))
        X = LSTMDataset(X)
        batch_size = 64
        dataloader = DataLoader(X, batch_size=batch_size, shuffle=True, num_workers=0)

        lstm_data_list = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for i, data in enumerate(dataloader):
            _, load = data
            load = load.to(device, dtype=torch.float)
            pre = self.lstm(load)
            lstm_data_list.append(pre)

        lstm_data = torch.concat(lstm_data_list, dim=0)

        rand_indx = torch.randperm(len(lstm_data))
        lstm_data = lstm_data[rand_indx]
        lstm_data = torch.transpose(lstm_data, 0, 1)
        lstm_data = lstm_data.reshape(-1, 8)

        lstm_data = pd.DataFrame(lstm_data.cpu().detach().numpy()).astype(np.float32)

        print("data generated...")
        if platform == "linux" or platform == "linux2":
            datafile = '/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/dataset/newRiver/lstmGeneratedData.csv'
        elif platform == "darwin":
            datafile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\lstmGeneratedData.csv'
        elif platform == "win32":
            datafile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\lstmGeneratedData.csv'
        lstm_data.to_csv(datafile)

    def savegroupdata(self):
        times = 1424//16
        # Generate 100 * 10 images and make a grid to save them.
        ygroup = []
        print("Gnerating as a Group...")
        for i in range(times):
            print("Iteration No. ", i)
            # single generation
            z_sample_single = Variable(torch.randn(self.n_output, self.z_dim)).cuda()
            imgs_group = (self.G(z_sample_single).data + 1.0) / 2.0
            loads_group, temp_group = Data.decode_16users(imgs_group)
            for group in loads_group:
                part = pd.DataFrame(group, dtype=np.float32)
                ygroup.append(part)
        Y_group = pd.concat(ygroup)
        print("data generated...")
        if platform == "linux" or platform == "linux2":
            datafile = '/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/dataset/newRiver/groupGeneratedData_Auto.csv'
        elif platform == "darwin":
            datafile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\groupGeneratedData.csv'
        elif platform == "win32":
            datafile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\groupGeneratedData.csv'
        Y_group.to_csv(datafile)

    def eval_with_data(self):
        times = 1424 // 16
        # real data
        real_data = Data.load_data()  # numpy  columns=8
        real_data = real_data.reshape((-1, self.npoint, self.nuser))
        real_data = np.random.permutation(real_data)[:self.n_output * times]

        # load lstm generated data
        if platform == "linux" or platform == "linux2":
            lstmdatafile = '~/Documents/Code/Research/MultiLoad-GAN_public/dataset/newRiver/lstmGeneratedData.csv'
        elif platform == "darwin":
            lstmdatafile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\lstmGeneratedData.csv'
        elif platform == "win32":
            lstmdatafile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\lstmGeneratedData.csv'
        data_lstm = pd.read_csv(lstmdatafile)
        gen_lstm = data_lstm.iloc[:, 1:].to_numpy()
        gen_lstm = gen_lstm.reshape((-1, self.npoint, self.nuser))
        # gen_lstm = gen_lstm[:320]
        gen_lstm = np.append(gen_lstm, gen_lstm, axis=0)
        # gen_lstm = np.append(gen_lstm, gen_lstm, axis=0)
        # gen_lstm = np.append(gen_lstm, gen_lstm[:144], axis=0)

        if platform == "linux" or platform == "linux2":
            # linux
            single_file = '/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/dataset/newRiver/singleGeneratedData.csv'
            group_file = '/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/dataset/newRiver/groupGeneratedData_Impressive.csv'  # W/O data augmentation
        elif platform == "darwin":
            # OS X
            single_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\singleGeneratedData.csv'
            group_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\groupGeneratedData.csv'
        elif platform == "win32":
            single_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\singleGeneratedData.csv'
            # group_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\groupGeneratedData_Impressive.csv'
            group_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\groupGeneratedData.csv'

        data_group = pd.read_csv(group_file)
        data_single = pd.read_csv(single_file)
        gen_group = data_group.iloc[:, 1:].to_numpy()
        gen_single = data_single.iloc[:, 1:].to_numpy()
        gen_group = gen_group.reshape((-1, self.npoint, self.nuser))
        # gen_group = np.append(gen_group, gen_group, axis=0)
        # gen_group = np.append(gen_group, gen_group, axis=0)
        # gen_group = np.append(gen_group, gen_group[:144], axis=0)
        gen_single = gen_single.reshape((-1, self.npoint, self.nuser))
        gen_single = np.append(gen_single, gen_single, axis=0)
        gen_single = np.append(gen_single, gen_single, axis=0)
        gen_single = np.append(gen_single, gen_single[:144], axis=0)

        print("data generated...")

        self.dispLoads_with_data(real_data, gen_group, gen_single, gen_lstm)
        self.rampAnalysis(real_data, torch.tensor(gen_group), torch.tensor(gen_single), torch.tensor(gen_lstm))
        self.dailyAnalysis(real_data, gen_group, gen_single, gen_lstm)
        self.hourlyAnalysis(real_data, gen_group, gen_single, gen_lstm)
        self.groupAnalysis(real_data, gen_group, gen_single, gen_lstm)

    def eval(self):
        times = 1424//16
        # real data
        real_data = Data.load_data()  # numpy  columns=8
        real_data = real_data.reshape((-1, self.npoint, self.nuser))
        real_data = np.random.permutation(real_data)[:self.n_output*times]

        # load lstm generated data
        if platform == "linux" or platform == "linux2":
            lstmdatafile = '~/Documents/Code/Research/MultiLoad-GAN_public/dataset/newRiver/lstmGeneratedData.csv'
        elif platform == "darwin":
            lstmdatafile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\lstmGeneratedData.csv'
        elif platform == "win32":
            lstmdatafile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\lstmGeneratedData.csv'
        data_lstm = pd.read_csv(lstmdatafile)
        gen_lstm = data_lstm.iloc[:, 1:].to_numpy()
        gen_lstm = gen_lstm.reshape((-1, self.npoint, self.nuser))
        gen_lstm = np.append(gen_lstm, gen_lstm, axis=0)

        # Generate 100 * 10 images and make a grid to save them.
        ygroup = []
        ysingle = []
        for i in range(times):
            # group generation
            print('Generating load group #', i)
            z_sample = Variable(torch.randn(self.n_output, self.z_dim)).cuda()
            imgs_sample = (self.G(z_sample).data + 1) / 2.0
            # decode imgs to loads  torch.tensor N * 96 * 8

            loads, temperature = Data.decode_16users(imgs_sample, self.npoint, self.nuser)
            ygroup.append(loads)

            # self.dispLoads(loads[0])
            # print(loads)

            # single generation
            z_sample_single = Variable(torch.randn(self.n_output*self.nuser, self.z_dim)).cuda()
            loads_single = self.singleUser_G(z_sample_single).data
            loads_single = loads_single * 5.0 + 5.0
            # print(loads_single)
            yloads = torch.zeros([self.n_output, self.npoint, self.nuser], dtype=torch.float32)
            for i in range(self.n_output):
                groupdata = []
                for c in range(self.nuser):
                    n = i * self.nuser + c
                    load = loads_single[n, 0].reshape((self.npoint, 1))
                    groupdata.append(load)
                group = torch.cat(groupdata, dim=1).cpu()    # 1 day 16 users    96*16
                part = pd.DataFrame(group).astype(np.float32)
                part = part.loc[:, part.mean().sort_values(ascending=True).index]
                group1 = torch.tensor(part.values)
                yloads[i] = group1
            ysingle.append(yloads)
            print('Dis single generated loads')
            # self.dispLoads(yloads[0])
        gen_group = torch.cat(ygroup, dim=0)
        gen_single = torch.cat(ysingle, dim=0)
        print("data generated...")


        self.dispLoads_with_data(real_data, gen_group.numpy(), gen_single.numpy())
        self.rampAnalysis(real_data, gen_group, gen_single)
        self.dailyAnalysis(real_data, gen_group.numpy(), gen_single.numpy())
        self.hourlyAnalysis(real_data, gen_group.numpy(), gen_single.numpy())
        self.groupAnalysis(real_data, gen_group.numpy(), gen_single.numpy())

    # input: numpy N*672*8
    def dispLoads_with_data(self, real_loads, group_loads, single_loads, lstm_loads):

        # ==========distributions================================
        week_real = []
        for n in range(real_loads.shape[0]):
            week = real_loads[n]
            week_real.append(week)
        np_real = np.concatenate(week_real, axis=1)

        week_single = []
        for n in range(single_loads.shape[0]):
            week = single_loads[n]
            week_single.append(week)
        np_single = np.concatenate(week_single, axis=1)

        week_group = []
        for n in range(group_loads.shape[0]):
            week = group_loads[n]
            week_group.append(week)
        np_group = np.concatenate(week_group, axis=1)

        week_lstm = []
        for n in range(lstm_loads.shape[0]):
            week = lstm_loads[n]
            week_lstm.append(week)
        np_lstm = np.concatenate(week_lstm, axis=1)

        df_real = pd.DataFrame(np_real)
        df_single = pd.DataFrame(np_single)
        df_group = pd.DataFrame(np_group)
        df_lstm = pd.DataFrame(np_lstm)

        mean_real = df_real.mean()
        max_real = df_real.max()
        mean_single = df_single.mean()
        max_single = df_single.max()
        mean_group = df_group.mean()
        max_group = df_group.max()
        mean_lstm = df_lstm.mean()
        max_lstm = df_lstm.max()

        mean2 = calculate_fid(mean_real.to_numpy(dtype=np.float32).reshape((-1, 1)), mean_single.to_numpy(dtype=np.float32).reshape((-1, 1)))
        mean1 = calculate_fid(mean_real.to_numpy(dtype=np.float32).reshape((-1, 1)),
                              mean_group.to_numpy(dtype=np.float32).reshape((-1, 1)))
        meanlstm = calculate_fid(mean_real.to_numpy(dtype=np.float32).reshape((-1, 1)),
                              mean_lstm.to_numpy(dtype=np.float32).reshape((-1, 1)))
        max2 = calculate_fid(max_real.to_numpy(dtype=np.float32).reshape((-1, 1)), max_single.to_numpy(dtype=np.float32).reshape((-1, 1)))
        max1 = calculate_fid(max_real.to_numpy(dtype=np.float32).reshape((-1, 1)),
                             max_group.to_numpy(dtype=np.float32).reshape((-1, 1)))
        maxlstm = calculate_fid(max_real.to_numpy(dtype=np.float32).reshape((-1, 1)),
                             max_lstm.to_numpy(dtype=np.float32).reshape((-1, 1)))
        print("FID for  mean ", mean1, mean2, meanlstm)
        print("FID for peak ", max1, max2, maxlstm)

        nb = 50
        plt.rc('font', size=12)  # controls default text sizes
        plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
        fig, axis = plt.subplots(2, 2, figsize=(fig_width, fig_height))
        sns.distplot(mean_real, bins=nb, hist=bhist, label='REAL', ax=axis[0,0], color='green')
        sns.distplot(mean_single, bins=nb, hist=bhist, label='SLGAN', ax=axis[0,0], color='orange')
        sns.distplot(mean_lstm, bins=nb, hist=bhist, label='LSTM', ax=axis[0, 0], color='red')
        sns.distplot(mean_group, bins=nb, hist=bhist, label='MLGAN', ax=axis[0,0], color='blue')
        axis[0, 0].set_ylabel('Density', labelpad=ylabel_distance)
        axis[0, 0].set_xticks([])
        axis[0, 0].set_yticks([])
        axis[0, 0].legend()
        sns.distplot(max_real, bins=nb, hist=bhist, label='REAL', ax=axis[0, 1], color='green')
        sns.distplot(max_single, bins=nb, hist=bhist, label='SLGAN', ax=axis[0, 1], color='orange')
        sns.distplot(max_lstm, bins=nb, hist=bhist, label='LSTM', ax=axis[0, 1], color='red')
        sns.distplot(max_group, bins=nb, hist=bhist, label='MLGAN', ax=axis[0, 1], color='blue')
        axis[0, 1].set_ylabel('', labelpad=ylabel_distance)
        axis[0, 1].set_xticks([])
        axis[0, 1].set_yticks([])
        axis[0, 1].legend()
        sns.boxplot(data=[mean_group, mean_single, mean_real, mean_lstm], orient='h', ax=axis[1, 0])
        axis[1, 0].set_yticklabels(['MLGAN', 'SLGAN', 'REAL', 'LSTM'])
        axis[1, 0].set_xlabel('kW', loc='right', labelpad=xlabel_distance)
        sns.boxplot(data=[max_group, max_single, max_real, max_lstm], orient='h', ax=axis[1, 1])
        axis[1, 1].set_yticklabels([])
        axis[1, 1].set_xlabel('kW', loc='right', labelpad=xlabel_distance)

        # axis[0, 0].set_title("household mean and peak")
        plt.subplots_adjust(wspace=0.1, hspace=0, top=top_margin, bottom=bottom_margin, left=left_margin, right=right_margin)
        plt.show()
        # fig.savefig('/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/figures/mean_peak_HL.svg', format="svg")

        # =========== only real ===============================
        fig, axis = plt.subplots(1, 2, figsize=(fig_width, 3))
        sns.distplot(mean_real, bins=nb, hist=bhist, label='REAL', ax=axis[0], color='green')
        axis[0].set_ylabel('Density', labelpad=ylabel_distance)
        axis[0].set_xlabel('kW', loc='right', labelpad=xlabel_distance)
        axis[0].set_yticks([])
        axis[0].legend()
        sns.distplot(max_real, bins=nb, hist=bhist, label='REAL', ax=axis[1], color='green')
        axis[1].set_ylabel('', labelpad=ylabel_distance)
        axis[1].set_yticks([])
        axis[1].legend()
        axis[1].set_xlabel('kW', loc='right', labelpad=xlabel_distance)
        # axis[0].set_title("household mean and peak")
        plt.subplots_adjust(wspace=0.1, hspace=0, top=top_margin, bottom=0.15, left=left_margin, right=right_margin)
        plt.show()
        # fig.savefig('/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/figures/mean_peak_onlyReal.svg', format="svg")
        # ========================================================

        real_loads = real_loads.reshape((-1, 672, 8))
        single_loads = single_loads.reshape((-1, 672, 8))
        group_loads = group_loads.reshape((-1, 672, 8))

        np.random.shuffle(real_loads)
        np.random.shuffle(single_loads)
        np.random.shuffle(group_loads)

        for i in range(10):
            fig, axis = plt.subplots(8, 3)
            for c in range(self.nuser):
                label = range(self.npoint)
                axis[c % 8, 0].plot(label, group_loads[i, :, c], color='blue')
                axis[c % 8, 0].set_ylim((0, 3))
                axis[c % 8, 0].set_xlabel('day', loc='right', labelpad=ylabel_distance)
                # axis[c % 8, 0].set_ylabel('kW', loc='top', labelpad=ylabel_distance)
                axis[c % 8, 0].set_yticks([])
                if c == 7:
                    axis[c % 8, 0].set_xticks(np.arange(0, 673, 96))
                    axis[c % 8, 0].set_xticklabels(np.arange(0, 8, 1))
                else:
                    axis[c % 8, 0].set_xticks([])

                axis[c % 8, 1].plot(label, real_loads[i, :, c], color='green')
                axis[c % 8, 1].set_ylim((0, 3))
                axis[c % 8, 1].set_xlabel('day', loc='right', labelpad=ylabel_distance)
                # axis[c % 8, 1].set_ylabel('kW', loc='top', labelpad=ylabel_distance)
                axis[c % 8, 1].set_yticks([])
                if c == 7:
                    axis[c % 8, 1].set_xticks(np.arange(0, 673, 96))
                    axis[c % 8, 1].set_xticklabels(np.arange(0, 8, 1))
                else:
                    axis[c % 8, 1].set_xticks([])

                axis[c % 8, 2].plot(label, single_loads[i, :, c], color='orange')
                axis[c % 8, 2].set_ylim((0, 3))
                axis[c % 8, 2].set_xlabel('day', loc='right', labelpad=ylabel_distance)
                # axis[c % 8, 2].set_ylabel('kW', loc='top', labelpad=ylabel_distance)
                axis[c % 8, 2].set_yticks([])
                if c == 7:
                    axis[c % 8, 2].set_xticks(np.arange(0, 673, 96))
                    axis[c % 8, 2].set_xticklabels(np.arange(0, 8, 1))
                else:
                    axis[c % 8, 2].set_xticks([])
            plt.subplots_adjust(wspace=0, hspace=0, top=top_margin, bottom=bottom_margin, left=0.05, right=right_margin)
            plt.show()
            # fig.savefig('/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/figures/generation_examples' + str(i) + '.svg', format="svg")

    def rampAnalysis(self, real_loads, group_loads, single_loads, lstm_loads):
        print('======rampAnalysis======')
        real_ramp = torch.zeros([real_loads.shape[0], self.npoint-1, self.nuser], dtype=torch.float32)
        group_ramp = torch.zeros([len(group_loads), self.npoint-1, self.nuser], dtype=torch.float32)
        single_ramp = torch.zeros([len(single_loads), self.npoint - 1, self.nuser], dtype=torch.float32)
        lstm_ramp = torch.zeros([len(lstm_loads), self.npoint - 1, self.nuser], dtype=torch.float32)
        for n in range(real_loads.shape[0]):
            real_ramp[n] = torch.from_numpy(real_loads[n, 1:, :] - real_loads[n, :-1, :])
        for m in range(group_loads.shape[0]):
            group_ramp[m] = group_loads[m, 1:, :] - group_loads[m, :-1, :]
        for k in range(single_loads.shape[0]):
            single_ramp[k] = single_loads[k, 1:, :] - single_loads[k, :-1, :]
        for l in range(lstm_loads.shape[0]):
            lstm_ramp[l] = lstm_loads[l, 1:, :] - lstm_loads[l, :-1, :]

        real_ramp = real_ramp.reshape((-1)).numpy()
        group_ramp = group_ramp.reshape((-1)).numpy()
        single_ramp = single_ramp.reshape((-1)).numpy()
        lstm_ramp = lstm_ramp.reshape((-1)).numpy()

        ramp2 = calculate_fid(real_ramp, single_ramp)
        ramp1 = calculate_fid(real_ramp, group_ramp)
        ramplstm = calculate_fid(real_ramp, lstm_ramp)
        print("FID for ramp ", ramp1, ramp2, ramplstm)

        nb = 40
        fig, axis = plt.subplots(1, 2, figsize=(fig_width, 3))
        sns.distplot(real_ramp, bins=nb, hist=bhist, label='REAL', color='green', ax=axis[1])
        sns.distplot(single_ramp, bins=nb, hist=bhist, label='SLGAN', color='orange', ax=axis[1])
        sns.distplot(lstm_ramp, bins=nb, hist=bhist, label='LSTM', color='red', ax=axis[1])
        sns.distplot(group_ramp, bins=nb, hist=bhist, label='MLGAN', color='blue', ax=axis[1])
        axis[1].set_xlabel('kW', loc='right', labelpad=xlabel_distance)
        axis[1].set_ylabel('Density', labelpad=ylabel_distance)
        axis[1].set_yticks([])
        axis[1].legend()

        sns.boxplot(data=[group_ramp, single_ramp, real_ramp, lstm_ramp], orient='h', ax=axis[0])
        axis[0].set_yticklabels(['MLGAN', 'SLGAN', 'REAL', 'LSTM'])
        axis[0].set_xlabel('kW', loc='right', labelpad=xlabel_distance)
        plt.subplots_adjust(wspace=0.1, hspace=0, top=top_margin, bottom=0.15, left=left_margin, right=right_margin)
        plt.show()
        # fig.savefig('/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/figures/ramp_HL.svg', format="svg")

    def hourlyAnalysis(self, real_loads, group_loads, single_loads, lstm_loads):
        real_loads_day = real_loads.reshape((-1, 96, 8))
        group_loads_day = group_loads.reshape((-1, 96, 8))
        single_loads_day = single_loads.reshape((-1, 96, 8))
        lstm_loads_day = lstm_loads.reshape((-1, 96, 8))

        day_real = []
        for n in range(real_loads_day.shape[0]):
            day = real_loads_day[n]
            day_real.append(day)
        np_real = np.concatenate(day_real, axis=1)  # 96*N

        day_single = []
        for n in range(single_loads_day.shape[0]):
            day = single_loads_day[n]
            day_single.append(day)
        np_single = np.concatenate(day_single, axis=1)

        day_lstm = []
        for n in range(lstm_loads_day.shape[0]):
            day = lstm_loads_day[n]
            day_lstm.append(day)
        np_lstm = np.concatenate(day_lstm, axis=1)

        day_group = []
        for n in range(group_loads_day.shape[0]):
            day = group_loads_day[n]
            day_group.append(day)
        np_group = np.concatenate(day_group, axis=1)

        np_real = np_real.reshape((24, 4, np_real.shape[1]))  # 24*4*N
        np_group = np_group.reshape((24, 4, np_group.shape[1]))
        np_single = np_single.reshape((24, 4, np_single.shape[1]))
        np_lstm = np_lstm.reshape((24, 4, np_lstm.shape[1]))

        nb = 100
        fig, axis = plt.subplots(6, 4)
        fig1, axis1 = plt.subplots(6, 4)
        for i in range(24):
            df_real = pd.DataFrame(np_real[i])  # 4*N
            df_group = pd.DataFrame(np_group[i])
            df_single = pd.DataFrame(np_single[i])
            df_lstm = pd.DataFrame(np_lstm[i])
            hourly_real = df_real.sum() / 4.0
            hourly_group = df_group.sum() / 4.0
            hourly_single = df_single.sum() / 4.0
            hourly_lstm = df_lstm.sum() / 4.0

            sns.distplot(hourly_real, bins=nb, hist=bhist, label='REAL', ax=axis[i%6, i//6], color='green')
            sns.distplot(hourly_single, bins=nb, hist=bhist, label='SLGAN', ax=axis[i%6, i//6], color='orange')
            sns.distplot(hourly_lstm, bins=nb, hist=bhist, label='LSTM', ax=axis[i % 6, i // 6], color='red')
            sns.distplot(hourly_group, bins=nb, hist=bhist, label='MLGAN', ax=axis[i%6, i//6], color='blue')
            if (i+1)%6 == 0:
                axis[i%6, i//6].set_xlabel('kWh', loc='right', labelpad=xlabel_distance)
            axis[i % 6, i // 6].set_ylabel('hour'+str(i+1), labelpad=ylabel_distance)
            axis[i%6, i//6].legend()

            sns.boxplot(data=[hourly_group, hourly_single, hourly_real, hourly_lstm], orient='h', ax=axis1[i % 6, i // 6])
            axis1[i % 6, i // 6].set_yticklabels(['MLGAN', 'SLGAN', 'REAL', 'LSTM'])
            if (i + 1) % 6 == 0:
                axis1[i % 6, i // 6].set_xlabel('kWh', loc='right', labelpad=xlabel_distance)
            # axis1[i % 6, i // 6].set_ylabel('hour' + str(i + 1), labelpad=ylabel_distance)
        plt.show()

        # =========== 6 hours =====================
        fig, axis = plt.subplots(4, 2, figsize=(fig_width, fig_height/2*3))
        hours = np.zeros((4, 3))
        for i in range(4):
            real_6hour_temp = np_real[i * 6: (i + 1) * 6]   # 6*4*N
            single_6hour_temp = np_single[i * 6: (i + 1) * 6]
            lstm_6hour_temp = np_lstm[i * 6: (i + 1) * 6]
            group_6hour_temp = np_group[i * 6: (i + 1) * 6]

            real_hour_temp = np.concatenate(real_6hour_temp, axis=1)
            single_hour_temp = np.concatenate(single_6hour_temp, axis=1)
            lstm_hour_temp = np.concatenate(lstm_6hour_temp, axis=1)
            group_hour_temp = np.concatenate(group_6hour_temp, axis=1)

            real_hourly_consumption = np.sum(real_hour_temp, axis=0) / 4.0
            single_hourly_consumption = np.sum(single_hour_temp, axis=0) / 4.0
            lstm_hourly_consumption = np.sum(lstm_hour_temp, axis=0) / 4.0
            group_hourly_consumption = np.sum(group_hour_temp, axis=0) / 4.0

            hours[i, 2] = calculate_fid(real_hourly_consumption, lstm_hourly_consumption)
            hours[i, 1] = calculate_fid(real_hourly_consumption, single_hourly_consumption)
            hours[i, 0] = calculate_fid(real_hourly_consumption, group_hourly_consumption)

            sns.distplot(real_hourly_consumption, bins=nb, hist=bhist, label='REAL', ax=axis[i % 2 * 2, i // 2],
                         color='green')
            sns.distplot(single_hourly_consumption, bins=nb, hist=bhist, label='SLGAN', ax=axis[i % 2 * 2, i // 2],
                         color='orange')
            sns.distplot(lstm_hourly_consumption, bins=nb, hist=bhist, label='LSTM', ax=axis[i % 2 * 2, i // 2],
                         color='red')
            sns.distplot(group_hourly_consumption, bins=nb, hist=bhist, label='MLGAN', ax=axis[i % 2 * 2, i // 2],
                         color='blue')
            axis[i % 2 * 2, i // 2].legend()

            sns.boxplot(data=[group_hourly_consumption, single_hourly_consumption, real_hourly_consumption, lstm_hourly_consumption], orient='h',
                        ax=axis[i % 2 * 2 + 1, i // 2])

            axis[i % 2 * 2 + 1, i // 2].set_yticklabels(['MLGAN', 'SLGAN', 'REAL', 'LSTM'])
            if i % 2 > 0:
                axis[i % 2 * 2 + 1, i // 2].set_xlabel('kWh', loc='right', labelpad=xlabel_distance)

        print("FID for hour ", hours)

        axis[0, 0].set_ylabel('1:00-6:00', labelpad=ylabel_distance)
        axis[0, 0].set_yticks([])
        axis[0, 1].set_ylabel('13:00-18:00', labelpad=ylabel_distance)
        axis[0, 1].set_yticks([])
        axis[2, 0].set_ylabel('7:00-12:00', labelpad=ylabel_distance)
        axis[2, 0].set_yticks([])
        axis[2, 1].set_ylabel('19:00-24:00', labelpad=ylabel_distance)
        axis[2, 1].set_yticks([])
        axis[1, 1].set_yticklabels([])
        axis[3, 1].set_yticklabels([])
        for i in range(3):
            for j in range(2):
                axis[i, j].set_xticks([])

        plt.subplots_adjust(wspace=0.1, hspace=0, top=top_margin, bottom=0.08, left=left_margin, right=right_margin)
        plt.show()
        # fig.savefig('/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/figures/hour_HL.svg', format="svg")

    # input numpy
    def dailyAnalysis(self, real_loads, group_loads, single_loads, lstm_loads):

        week_real = []
        for n in range(real_loads.shape[0]):
            week = real_loads[n]
            week_real.append(week)
        np_real = np.concatenate(week_real, axis=1)     # 672*N

        week_single = []
        for n in range(single_loads.shape[0]):
            week = single_loads[n]
            week_single.append(week)
        np_single = np.concatenate(week_single, axis=1)

        week_lstm = []
        for n in range(lstm_loads.shape[0]):
            week = lstm_loads[n]
            week_lstm.append(week)
        np_lstm = np.concatenate(week_lstm, axis=1)

        week_group = []
        for n in range(group_loads.shape[0]):
            week = group_loads[n]
            week_group.append(week)
        np_group = np.concatenate(week_group, axis=1)

        np_real = np_real.reshape((7, 96, np_real.shape[1]))    # 7*96*N
        np_group = np_group.reshape((7, 96, np_group.shape[1]))
        np_single = np_single.reshape((7, 96, np_single.shape[1]))
        np_lstm = np_lstm.reshape((7, 96, np_lstm.shape[1]))

        nb = 25
        fig, axis = plt.subplots(2, 7)

        day = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']
        for i in range(7):
            df_real = pd.DataFrame(np_real[i])      # 96*N
            df_group = pd.DataFrame(np_group[i])
            df_single = pd.DataFrame(np_single[i])
            df_lstm = pd.DataFrame(np_lstm[i])
            daily_real = df_real.sum() / 4.0
            daily_group = df_group.sum() / 4.0
            daily_single = df_single.sum() / 4.0
            daily_lstm = df_lstm.sum() / 4.0

            sns.distplot(daily_real, bins=nb, hist=bhist, label='REAL', ax=axis[0, i],
                         color='green')
            sns.distplot(daily_single, bins=nb, hist=bhist, label='SLGAN', ax=axis[0, i], color='orange')
            sns.distplot(daily_lstm, bins=nb, hist=bhist, label='LSTM', ax=axis[0, i], color='red')
            sns.distplot(daily_group, bins=nb, hist=bhist, label='MLGAN', ax=axis[0, i], color='blue')
            axis[0, i].set_ylabel('Density', labelpad=ylabel_distance)
            # axis[0, i].set_xlabel('kWh', loc='right', labelpad=xlabel_distance)
            axis[0, i].legend()

            sns.boxplot(data=[daily_group, daily_single, daily_real, daily_lstm], orient='h', ax=axis[1, i])
            axis[1, i].set_yticklabels(['MLGAN', 'SLGAN', 'REAL', 'LSTM'])
            axis[1, i].set_xlabel('kWh', loc='right', labelpad=xlabel_distance)
        plt.show()

        # ========= weekday and weekend ======================
        fig, axis = plt.subplots(2, 2, figsize=(fig_width, fig_height))
        real_weekday_consumption = np.array([], dtype=np.float32)
        single_weekday_consumption = np.array([], dtype=np.float32)
        lstm_weekday_consumption = np.array([], dtype=np.float32)
        group_weekday_consumption = np.array([], dtype=np.float32)
        for i in range(5):
            real_day_temp = np_real[i]     # 96*N
            single_day_temp = np_single[i]
            lstm_day_temp = np_lstm[i]
            group_day_temp = np_group[i]
            real_daily_consumption = np.sum(real_day_temp, axis=0) / 4.0
            single_daily_consumption = np.sum(single_day_temp, axis=0) / 4.0
            lstm_daily_consumption = np.sum(lstm_day_temp, axis=0) / 4.0
            group_daily_consumption = np.sum(group_day_temp, axis=0) / 4.0

            real_weekday_consumption = np.append(real_weekday_consumption, real_daily_consumption)
            single_weekday_consumption = np.append(single_weekday_consumption, single_daily_consumption)
            lstm_weekday_consumption = np.append(lstm_weekday_consumption, lstm_daily_consumption)
            group_weekday_consumption = np.append(group_weekday_consumption, group_daily_consumption)

        sns.distplot(real_weekday_consumption, bins=nb, hist=bhist, label='REAL', ax=axis[0, 0], color='green')
        sns.distplot(single_weekday_consumption, bins=nb, hist=bhist, label='SLGAN', ax=axis[0, 0], color='orange')
        sns.distplot(lstm_weekday_consumption, bins=nb, hist=bhist, label='LSTM', ax=axis[0, 0], color='red')
        sns.distplot(group_weekday_consumption, bins=nb, hist=bhist, label='MLGAN', ax=axis[0, 0], color='blue')
        axis[0, 0].set_ylabel('Density', labelpad=ylabel_distance)
        axis[0, 0].set_xticks([])
        axis[0, 0].set_yticks([])
        axis[0, 0].legend()
        sns.boxplot(data=[group_weekday_consumption, single_weekday_consumption, real_weekday_consumption, lstm_weekday_consumption], orient='h',
                    ax=axis[1, 0])
        axis[1, 0].set_yticklabels(['MLGAN', 'SLGAN', 'REAL', 'LSTM'])
        axis[1, 0].set_xlabel('kWh', loc='right', labelpad=xlabel_distance)

        real_weekend_consumption = np.array([], dtype=np.float32)
        single_weekend_consumption = np.array([], dtype=np.float32)
        lstm_weekend_consumption = np.array([], dtype=np.float32)
        group_weekend_consumption = np.array([], dtype=np.float32)
        for i in range(5, 7):
            real_day_temp = np_real[i]  # 96*N
            single_day_temp = np_single[i]
            lstm_day_temp = np_lstm[i]
            group_day_temp = np_group[i]
            real_daily_consumption = np.sum(real_day_temp, axis=0) / 4.0
            single_daily_consumption = np.sum(single_day_temp, axis=0) / 4.0
            lstm_daily_consumption = np.sum(lstm_day_temp, axis=0) / 4.0
            group_daily_consumption = np.sum(group_day_temp, axis=0) / 4.0

            real_weekend_consumption = np.append(real_weekend_consumption, real_daily_consumption)
            single_weekend_consumption = np.append(single_weekend_consumption, single_daily_consumption)
            lstm_weekend_consumption = np.append(lstm_weekend_consumption, lstm_daily_consumption)
            group_weekend_consumption = np.append(group_weekend_consumption, group_daily_consumption)

        sns.distplot(real_weekend_consumption, bins=nb, hist=bhist, label='REAL', ax=axis[0, 1], color='green')
        sns.distplot(single_weekend_consumption, bins=nb, hist=bhist, label='SLGAN', ax=axis[0, 1], color='orange')
        sns.distplot(lstm_weekend_consumption, bins=nb, hist=bhist, label='LSTM', ax=axis[0, 1], color='red')
        sns.distplot(group_weekend_consumption, bins=nb, hist=bhist, label='MLGAN', ax=axis[0, 1], color='blue')
        axis[0, 1].set_ylabel('', labelpad=ylabel_distance)
        axis[0, 1].set_xticks([])
        axis[0, 1].set_yticks([])
        axis[0, 1].legend()
        sns.boxplot(data=[group_weekend_consumption, single_weekend_consumption, real_weekend_consumption, lstm_weekend_consumption], orient='h',
                    ax=axis[1, 1])
        axis[1, 1].set_yticklabels([])
        axis[1, 1].set_xlabel('kWh', loc='right', labelpad=xlabel_distance)
        # axis[0, 0].set_title("household weekday and weekend")
        plt.subplots_adjust(wspace=0.1, hspace=0, top=top_margin, bottom=bottom_margin, left=left_margin, right=right_margin)
        plt.show()
        # fig.savefig('/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/figures/day_HL.svg', format="svg")

        weekday2 = calculate_fid(real_weekday_consumption, single_weekday_consumption)
        weekday1 = calculate_fid(real_weekday_consumption, group_weekday_consumption)
        weekdaylstm = calculate_fid(real_weekday_consumption, lstm_weekday_consumption)
        weekend2 = calculate_fid(real_weekend_consumption, single_weekend_consumption)
        weekend1 = calculate_fid(real_weekend_consumption, group_weekend_consumption)
        weekendlstm = calculate_fid(real_weekend_consumption, lstm_weekend_consumption)
        print("FID for weekday ", weekday1, weekday2, weekdaylstm)
        print("FID for weekend ", weekend1, weekend2, weekendlstm)

    def groupAnalysis(self, real_loads, group_loads, single_loads, lstm_loads):
        # add 16 users together
        real_agg = np.empty([real_loads.shape[0], self.npoint], dtype=np.float32)        # N*672
        group_agg = np.empty([group_loads.shape[0], self.npoint], dtype=np.float32)
        single_agg = np.empty([single_loads.shape[0], self.npoint], dtype=np.float32)
        lstm_agg = np.empty([lstm_loads.shape[0], self.npoint], dtype=np.float32)

        for n in range(real_loads.shape[0]):
            for i in range(self.npoint):
                real_agg[n, i] = np.sum(real_loads[n, i, :])
        for n in range(group_loads.shape[0]):
            for i in range(self.npoint):
                group_agg[n, i] = np.sum(group_loads[n, i, :])
        for n in range(single_loads.shape[0]):
            for i in range(self.npoint):
                single_agg[n, i] = np.sum(single_loads[n, i, :])
        for n in range(lstm_loads.shape[0]):
            for i in range(self.npoint):
                lstm_agg[n, i] = np.sum(lstm_loads[n, i, :])

        real_day = real_agg.reshape((-1, 96))       # 7*N*96
        group_day = group_agg.reshape((-1, 96))
        single_day = single_agg.reshape((-1, 96))
        lstm_day = lstm_agg.reshape((-1, 96))

        # ===================mean and peak and peak time===================
        real_means, real_maxs, real_peaktime = np.mean(real_day, axis=1), np.max(real_day, axis=1), np.argmax(real_day, axis=1)
        group_means, group_maxs, group_peaktime = np.mean(group_day, axis=1), np.max(group_day, axis=1), np.argmax(group_day, axis=1)
        single_means, single_maxs, single_peaktime = np.mean(single_day, axis=1), np.max(single_day, axis=1), np.argmax(single_day, axis=1)
        lstm_means, lstm_maxs, lstm_peaktime = np.mean(lstm_day, axis=1), np.max(lstm_day, axis=1), np.argmax(
            lstm_day, axis=1)

        meanlstm = calculate_fid(real_means, lstm_means)
        mean2 = calculate_fid(real_means, single_means)
        mean1 = calculate_fid(real_means, group_means)
        maxlstm = calculate_fid(real_maxs, lstm_maxs)
        max2 = calculate_fid(real_maxs, single_maxs)
        max1 = calculate_fid(real_maxs, group_maxs)
        print("FID for mean ", mean1, mean2, meanlstm)
        print("FID for peak", max1, max2, maxlstm)

        nb = 50
        fig, axis = plt.subplots(2, 2, figsize=(fig_width, fig_height))
        sns.distplot(real_means, bins=nb, hist=bhist, label='REAL', ax=axis[0, 0], color='green')
        sns.distplot(single_means, bins=nb, hist=bhist, label='SLGAN', ax=axis[0, 0], color='orange')
        sns.distplot(lstm_means, bins=nb, hist=bhist, label='LSTM', ax=axis[0, 0], color='red')
        sns.distplot(group_means, bins=nb, hist=bhist, label='MLGAN', ax=axis[0, 0], color='blue')
        axis[0, 0].set_ylabel('Density', labelpad=ylabel_distance)
        axis[0, 0].set_xticks([])
        axis[0, 0].set_yticks([])
        axis[0, 0].legend()
        sns.distplot(real_maxs, bins=nb, hist=bhist, label='REAL', ax=axis[0, 1], color='green')
        sns.distplot(single_maxs, bins=nb, hist=bhist, label='SLGAN', ax=axis[0, 1], color='orange')
        sns.distplot(lstm_maxs, bins=nb, hist=bhist, label='LSTM', ax=axis[0, 1], color='red')
        sns.distplot(group_maxs, bins=nb, hist=bhist, label='MLGAN', ax=axis[0, 1], color='blue')
        axis[0, 1].set_ylabel('', labelpad=ylabel_distance)
        axis[0, 1].set_xticks([])
        axis[0, 1].set_yticks([])
        axis[0, 1].legend()

        sns.boxplot(data=[group_means, single_means, real_means, lstm_means], orient='h', ax=axis[1, 0])
        axis[1, 0].set_yticklabels(['MLGAN', 'SLGAN', 'REAL', 'LSTM'])
        axis[1, 0].set_xlabel('kW', loc='right', labelpad=xlabel_distance)
        sns.boxplot(data=[group_maxs, single_maxs, real_maxs, lstm_maxs], orient='h', ax=axis[1, 1])
        axis[1, 1].set_yticklabels([])
        axis[1, 1].set_xlabel('kW', loc='right', labelpad=xlabel_distance)

        plt.subplots_adjust(wspace=0.1, hspace=0, top=top_margin, bottom=bottom_margin, left=left_margin, right=right_margin)
        plt.show()
        # fig.savefig('/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/figures/mean_peak_AL.svg', format="svg")

        # ===========ramping==========================
        real_ramp = real_agg[:, 1:] - real_agg [:, :-1]
        group_ramp = group_agg[:, 1:] - group_agg[:, :-1]
        single_ramp = single_agg[:, 1:] - single_agg[:, :-1]
        lstm_ramp = lstm_agg[:, 1:] - lstm_agg[:, :-1]
        real_ramp = real_ramp.reshape((-1))
        single_ramp = single_ramp.reshape((-1))
        lstm_ramp = lstm_ramp.reshape((-1))
        group_ramp = group_ramp.reshape((-1))

        ramplstm = calculate_fid(real_ramp, lstm_ramp)
        ramp2 = calculate_fid(real_ramp, single_ramp)
        ramp1 = calculate_fid(real_ramp, group_ramp)
        print("FID for ramp ", ramp1, ramp2, ramplstm)

        nb = 40
        fig, axis = plt.subplots(1, 2, figsize=(fig_width, 3))
        sns.distplot(real_ramp, bins=nb, hist=bhist, label='REAL', color='green', ax=axis[1])
        sns.distplot(single_ramp, bins=nb, hist=bhist, label='SLGAN', color='orange', ax=axis[1])
        sns.distplot(lstm_ramp, bins=nb, hist=bhist, label='LSTM', color='red', ax=axis[1])
        sns.distplot(group_ramp, bins=nb, hist=bhist, label='MLGAN', color='blue', ax=axis[1])
        axis[1].set_xlabel('kW', loc='right', labelpad=xlabel_distance)
        axis[1].set_ylabel('Density', labelpad=ylabel_distance)
        axis[1].set_yticks([])
        axis[1].legend()
        sns.boxplot(data=[group_ramp, single_ramp, real_ramp, lstm_ramp], orient='h',
                    ax=axis[0])
        axis[0].set_yticklabels(['MLGAN', 'SLGAN', 'REAL', 'LSTM'])
        axis[0].set_xlabel('kW', loc='right', labelpad=xlabel_distance)
        # plt.title("Agg Ramping Distribution")
        plt.subplots_adjust(wspace=0.1, hspace=0, top=top_margin, bottom=0.15, left=left_margin, right=right_margin)
        plt.show()
        # fig.savefig('/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/figures/ramp_AL.svg', format="svg")

# ========== daily consumption=========================
        nb = 25
        day = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']
        fig, axis = plt.subplots(2, 7)
        for i in range(7):
            real_day_temp = real_agg[:, i * 96:(i+1) * 96]
            single_day_temp = single_agg[:, i * 96:(i + 1) * 96]
            lstm_day_temp = lstm_agg[:, i * 96:(i + 1) * 96]
            group_day_temp = group_agg[:, i * 96:(i + 1) * 96]
            real_daily_consumption = np.sum(real_day_temp, axis=1) / 4.0
            single_daily_consumption = np.sum(single_day_temp, axis=1) / 4.0
            lstm_daily_consumption = np.sum(lstm_day_temp, axis=1) / 4.0
            group_daily_consumption = np.sum(group_day_temp, axis=1) / 4.0

            sns.distplot(real_daily_consumption, bins=nb, hist=bhist, label='REAL', ax=axis[0, i],
                         color='green')
            sns.distplot(single_daily_consumption, bins=nb, hist=bhist, label='SLGAN', ax=axis[0, i], color='orange')
            sns.distplot(lstm_daily_consumption, bins=nb, hist=bhist, label='LSTM', ax=axis[0, i], color='red')
            sns.distplot(group_daily_consumption, bins=nb, hist=bhist, label='MLGAN', ax=axis[0, i], color='blue')
            axis[0, i].set_ylabel('Density', labelpad=ylabel_distance)
            axis[0, i].legend()

            sns.boxplot(data=[group_daily_consumption, single_daily_consumption, real_daily_consumption, lstm_daily_consumption], orient='h', ax=axis[1, i])
            axis[1, i].set_yticklabels(['MLGAN', 'SLGAN', 'REAL', 'LSTM'])
            axis[1, i].set_xlabel('kWh', loc='right', labelpad=xlabel_distance)
        plt.show()

    # ========= weekday and weekend ======================
        fig, axis = plt.subplots(2, 2, figsize=(fig_width, fig_height))
        real_weekday_consumption = np.array([], dtype=np.float32)
        single_weekday_consumption = np.array([], dtype=np.float32)
        lstm_weekday_consumption = np.array([], dtype=np.float32)
        group_weekday_consumption = np.array([], dtype=np.float32)
        for i in range(5):
            real_day_temp = real_agg[:, i * 96:(i + 1) * 96]
            single_day_temp = single_agg[:, i * 96:(i + 1) * 96]
            lstm_day_temp = lstm_agg[:, i * 96:(i + 1) * 96]
            group_day_temp = group_agg[:, i * 96:(i + 1) * 96]
            real_daily_consumption = np.sum(real_day_temp, axis=1) / 4.0
            single_daily_consumption = np.sum(single_day_temp, axis=1) / 4.0
            lstm_daily_consumption = np.sum(lstm_day_temp, axis=1) / 4.0
            group_daily_consumption = np.sum(group_day_temp, axis=1) / 4.0

            real_weekday_consumption = np.append(real_weekday_consumption, real_daily_consumption)
            single_weekday_consumption = np.append(single_weekday_consumption, single_daily_consumption)
            lstm_weekday_consumption = np.append(lstm_weekday_consumption, lstm_daily_consumption)
            group_weekday_consumption = np.append(group_weekday_consumption, group_daily_consumption)

        sns.distplot(real_weekday_consumption, bins=nb, hist=bhist, label='REAL', ax=axis[0, 0], color='green')
        sns.distplot(single_weekday_consumption, bins=nb, hist=bhist, label='SLGAN', ax=axis[0, 0], color='orange')
        sns.distplot(lstm_weekday_consumption, bins=nb, hist=bhist, label='LSTM', ax=axis[0, 0], color='red')
        sns.distplot(group_weekday_consumption, bins=nb, hist=bhist, label='MLGAN', ax=axis[0, 0], color='blue')
        axis[0, 0].set_ylabel('Density', labelpad=ylabel_distance)
        axis[0, 0].set_xticks([])
        axis[0, 0].set_yticks([])
        axis[0, 0].legend()
        sns.boxplot(data=[group_weekday_consumption, single_weekday_consumption, real_weekday_consumption, lstm_weekday_consumption], orient='h', ax=axis[1, 0])
        axis[1, 0].set_yticklabels(['MLGAN', 'SLGAN', 'REAL', 'LSTM'])
        axis[1, 0].set_xlabel('kWh', loc='right', labelpad=xlabel_distance)

        real_weekend_consumption = np.array([], dtype=np.float32)
        single_weekend_consumption = np.array([], dtype=np.float32)
        lstm_weekend_consumption = np.array([], dtype=np.float32)
        group_weekend_consumption = np.array([], dtype=np.float32)
        for i in range(5, 7):
            real_day_temp = real_agg[:, i * 96:(i + 1) * 96]
            single_day_temp = single_agg[:, i * 96:(i + 1) * 96]
            lstm_day_temp = lstm_agg[:, i * 96:(i + 1) * 96]
            group_day_temp = group_agg[:, i * 96:(i + 1) * 96]
            real_daily_consumption = np.sum(real_day_temp, axis=1) / 4.0
            single_daily_consumption = np.sum(single_day_temp, axis=1) / 4.0
            lstm_daily_consumption = np.sum(lstm_day_temp, axis=1) / 4.0
            group_daily_consumption = np.sum(group_day_temp, axis=1) / 4.0

            real_weekend_consumption = np.append(real_weekend_consumption, real_daily_consumption)
            single_weekend_consumption = np.append(single_weekend_consumption, single_daily_consumption)
            lstm_weekend_consumption = np.append(lstm_weekend_consumption, lstm_daily_consumption)
            group_weekend_consumption = np.append(group_weekend_consumption, group_daily_consumption)

        sns.distplot(real_weekend_consumption, bins=nb, hist=bhist, label='REAL', ax=axis[0, 1], color='green')
        sns.distplot(single_weekend_consumption, bins=nb, hist=bhist, label='SLGAN', ax=axis[0, 1], color='orange')
        sns.distplot(lstm_weekend_consumption, bins=nb, hist=bhist, label='LSTM', ax=axis[0, 1], color='red')
        sns.distplot(group_weekend_consumption, bins=nb, hist=bhist, label='MLGAN', ax=axis[0, 1], color='blue')
        axis[0, 1].set_ylabel('', labelpad=ylabel_distance)
        axis[0, 1].set_xticks([])
        axis[0, 1].set_yticks([])
        axis[0, 1].legend()
        sns.boxplot(data=[group_weekend_consumption, single_weekend_consumption, real_weekend_consumption, lstm_weekend_consumption], orient='h',
                    ax=axis[1, 1])
        axis[1, 1].set_yticklabels([])
        axis[1, 1].set_xlabel('kWh', loc='right', labelpad=xlabel_distance)
        # axis[0, 0].set_title("Agg weekday and weekend")
        plt.subplots_adjust(wspace=0.1, hspace=0, top=top_margin, bottom=bottom_margin, left=left_margin, right=right_margin)
        plt.show()
        # fig.savefig('/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/figures/day_AL.svg', format="svg")

        weekdaylstm = calculate_fid(real_weekday_consumption, lstm_weekday_consumption)
        weekday2 = calculate_fid(real_weekday_consumption, single_weekday_consumption)
        weekday1 = calculate_fid(real_weekday_consumption, group_weekday_consumption)
        weekendlstm = calculate_fid(real_weekend_consumption, lstm_weekend_consumption)
        weekend2 = calculate_fid(real_weekend_consumption, single_weekend_consumption)
        weekend1 = calculate_fid(real_weekend_consumption, group_weekend_consumption)
        print("FID for weekday ", weekday1, weekday2, weekdaylstm)
        print("FID for weekend ", weekend1, weekend2, weekendlstm)

# ================= hourly consumption ======================
        nb = 100
        fig, axis = plt.subplots(6, 4)
        fig1, axis1 = plt.subplots(6, 4)
        for i in range(24):
            real_hour_temp = real_day[:, i * 4:(i + 1) * 4]
            single_hour_temp = single_day[:, i * 4:(i + 1) * 4]
            lstm_hour_temp = lstm_day[:, i * 4:(i + 1) * 4]
            group_hour_temp = group_day[:, i * 4:(i + 1) * 4]
            real_hourly_consumption = np.sum(real_hour_temp, axis=1) / 4.0
            single_hourly_consumption = np.sum(single_hour_temp, axis=1) / 4.0
            lstm_hourly_consumption = np.sum(lstm_hour_temp, axis=1) / 4.0
            group_hourly_consumption = np.sum(group_hour_temp, axis=1) / 4.0

            sns.distplot(real_hourly_consumption, bins=nb, hist=bhist, label='REAL', ax=axis[i%6, i//6],
                         color='green')
            sns.distplot(single_hourly_consumption, bins=nb, hist=bhist, label='SLGAN', ax=axis[i%6, i//6], color='orange')
            sns.distplot(lstm_hourly_consumption, bins=nb, hist=bhist, label='LSTM', ax=axis[i % 6, i // 6],
                         color='red')
            sns.distplot(group_hourly_consumption, bins=nb, hist=bhist, label='MLGAN', ax=axis[i%6, i//6], color='blue')

            if (i+1)%6 == 0:
                axis[i%6, i//6].set_xlabel('kWh', loc='right', labelpad=xlabel_distance)
            axis[i % 6, i // 6].set_ylabel('hour'+str(i+1), labelpad=ylabel_distance)
            axis[i%6, i//6].legend()

            sns.boxplot(data=[group_hourly_consumption, single_hourly_consumption, real_hourly_consumption, lstm_hourly_consumption], orient='h', ax=axis1[i % 6, i // 6])
            axis1[i % 6, i // 6].set_yticklabels(['MLGAN', 'SLGAN', 'REAL', 'LSTM'])
            if (i + 1) % 6 == 0:
                axis1[i % 6, i // 6].set_xlabel('kWh', loc='right', labelpad=xlabel_distance)
            # axis1[i % 6, i // 6].set_ylabel('hour' + str(i + 1), labelpad=ylabel_distance)
        plt.show()

    # =========== 6 hours =====================
        fig, axis = plt.subplots(4, 2, figsize=(fig_width, fig_height/2*3))
        hours = np.zeros((4, 3))
        for i in range(4):
            real_6hour_temp = real_day[:, i * 24:(i + 1) * 24]
            single_6hour_temp = single_day[:, i * 24:(i + 1) * 24]
            lstm_6hour_temp = lstm_day[:, i * 24:(i + 1) * 24]
            group_6hour_temp = group_day[:, i * 24:(i + 1) * 24]
            real_hour_temp = real_6hour_temp.reshape((-1, 4))
            single_hour_temp = single_6hour_temp.reshape((-1, 4))
            lstm_hour_temp = lstm_6hour_temp.reshape((-1, 4))
            group_hour_temp = group_6hour_temp.reshape((-1, 4))
            real_hourly_consumption = np.sum(real_hour_temp, axis=1) / 4.0
            single_hourly_consumption = np.sum(single_hour_temp, axis=1) / 4.0
            lstm_hourly_consumption = np.sum(lstm_hour_temp, axis=1) / 4.0
            group_hourly_consumption = np.sum(group_hour_temp, axis=1) / 4.0

            hours[i, 2] = calculate_fid(real_hourly_consumption, lstm_hourly_consumption)
            hours[i, 1] = calculate_fid(real_hourly_consumption, single_hourly_consumption)
            hours[i, 0] = calculate_fid(real_hourly_consumption, group_hourly_consumption)

            sns.distplot(real_hourly_consumption, bins=nb, hist=bhist, label='REAL', ax=axis[i % 2 * 2, i // 2], color='green')
            sns.distplot(single_hourly_consumption, bins=nb, hist=bhist, label='SLGAN', ax=axis[i % 2 * 2, i // 2], color='orange')
            sns.distplot(lstm_hourly_consumption, bins=nb, hist=bhist, label='LSTM', ax=axis[i % 2 * 2, i // 2],
                         color='red')
            sns.distplot(group_hourly_consumption, bins=nb, hist=bhist, label='MLGAN', ax=axis[i % 2 * 2, i // 2], color='blue')
            axis[i % 2 * 2, i // 2].legend()

            sns.boxplot(data=[group_hourly_consumption, single_hourly_consumption, real_hourly_consumption, lstm_hourly_consumption], orient='h',
                        ax=axis[i % 2 * 2 + 1, i // 2])
            axis[i % 2 * 2 + 1, i // 2].set_yticklabels(['MLGAN', 'SLGAN', 'REAL', 'LSTM'])
            if i % 2 > 0:
                axis[i % 2 * 2 + 1, i // 2].set_xlabel('kWh', loc='right', labelpad=xlabel_distance)

        print("FID for hour ", hours)

        axis[0, 0].set_ylabel('1:00-6:00', labelpad=ylabel_distance)
        axis[0, 0].set_yticks([])
        axis[0, 1].set_ylabel('13:00-18:00', labelpad=ylabel_distance)
        axis[0, 1].set_yticks([])
        axis[2, 0].set_ylabel('7:00-12:00', labelpad=ylabel_distance)
        axis[2, 0].set_yticks([])
        axis[2, 1].set_ylabel('19:00-24:00', labelpad=ylabel_distance)
        axis[2, 1].set_yticks([])
        # axis[0, 0].set_title("Agg hours")
        axis[1, 1].set_yticklabels([])
        axis[3, 1].set_yticklabels([])
        for i in range(3):
            for j in range(2):
                axis[i, j].set_xticks([])

        plt.subplots_adjust(wspace=0.1, hspace=0, top=top_margin, bottom=0.08, left=left_margin, right=right_margin)
        plt.show()
        # fig.savefig('/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/figures/hour_AL.svg', format="svg")

    # Not used
    # display one day load  96*7*8
    def dispLoads(self):
        if platform == "linux" or platform == "linux2":
            real_file = '/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/dataset/newRiver/GANData0.csv'
            single_file = '/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/dataset/newRiver/singleGeneratedData.csv'
            group_file = '/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/dataset/newRiver/groupGeneratedData.csv'
        elif platform == "darwin":
            real_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\GANData0.csv'
            single_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\singleGeneratedData.csv'
            group_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\groupGeneratedData.csv'
        elif platform == "win32":
            real_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\GANData0.csv'
            single_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\singleGeneratedData.csv'
            group_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\groupGeneratedData.csv'
        real_data = pd.read_csv(real_file)
        single_data = pd.read_csv(single_file)
        group_data = pd.read_csv(group_file)

        real_loads = real_data.iloc[:, 1:-1].to_numpy()
        single_loads = single_data.iloc[:, 1:].to_numpy()
        group_loads = group_data.iloc[:, 1:].to_numpy()

        # ==========distributions================================
        week_real = []
        for n in range(len(real_data.index) // 672):
            week = real_data.iloc[n * 672:n * 672 + 672, 1:-1]
            week.index = range(len(week.index))
            week_real.append(week)
        df_real = pd.concat(week_real, axis=1)

        week_single = []
        for n in range(len(single_data.index) // 672):
            week = single_data.iloc[n * 672:n * 672 + 672, 1:]
            week.index = range(len(week.index))
            week_single.append(week)
        df_single = pd.concat(week_single, axis=1)

        week_group = []
        for n in range(len(group_data.index) // 672):
            week = group_data.iloc[n * 672:n * 672 + 672, 1:]
            week.index = range(len(week.index))
            week_group.append(week)
        df_group = pd.concat(week_group, axis=1)

        mean_real = df_real.mean()
        max_real = df_real.max()
        mean_single = df_single.mean()
        max_single = df_single.max()
        mean_group = df_group.mean()
        max_group = df_group.max()

        nb = 50
        fig, axis = plt.subplots(1, 2, figsize=(fig_width, fig_height/2))
        axis[0].hist(mean_real, bins=nb, density=True, histtype='step', label='REAL', color='green')
        axis[0].hist(mean_single, bins=nb, range=(mean_real.min(), mean_real.max()), density=True, histtype='step',
                     label='single', color='orange')
        axis[0].hist(mean_group, bins=nb, range=(mean_real.min(), mean_real.max()), density=True, histtype='step',
                     label='group', color='blue')
        axis[0].set_title('mean distribution')
        axis[0].legend()
        axis[1].hist(max_real, bins=nb, density=True, histtype='step', label='REAL', color='green')
        axis[1].hist(max_single, bins=nb, range=(max_real.min(), max_real.max()), density=True, histtype='step',
                     label='single', color='orange')
        axis[1].hist(max_group, bins=nb, range=(max_real.min(), max_real.max()), density=True, histtype='step',
                     label='group', color='blue')
        axis[1].set_title('peak distribution')
        axis[1].legend()
        plt.show()
        # ========================================================

        real_loads = real_loads.reshape((-1, 672, 8))
        single_loads = single_loads.reshape((-1, 672, 8))
        group_loads = group_loads.reshape((-1, 672, 8))

        np.random.shuffle(real_loads)
        np.random.shuffle(single_loads)
        np.random.shuffle(group_loads)

        for i in range(10):
            fig, axis = plt.subplots(8, 3)
            for c in range(self.nuser):
                label = range(self.npoint)
                axis[c % 8, 0].plot(label, group_loads[i, :, c], color='blue')
                axis[c % 8, 0].set_ylim((0, 3))
                axis[c % 8, 0].set_xlabel('t')
                axis[c % 8, 0].set_ylabel('kW')

                axis[c % 8, 1].plot(label, real_loads[i, :, c], color='green')
                axis[c % 8, 1].set_ylim((0, 3))
                axis[c % 8, 1].set_xlabel('t')
                axis[c % 8, 1].set_ylabel('kW')

                axis[c % 8, 2].plot(label, single_loads[i, :, c], color='orange')
                axis[c % 8, 2].set_ylim((0, 3))
                axis[c % 8, 2].set_xlabel('t')
                axis[c % 8, 2].set_ylabel('kW')
            plt.show()

    # Not used
    def statTD(self, Temp, DOW):
        for n in range(Temp.shape[0]):
            D = DOW[n].reshape((-1,))
            T = torch.zeros([2, 96])
            torch.set_printoptions(profile="full")
            print(D)
            for i in range(96):
                T[0, i] = torch.mean(Temp[n, i, :])
                T[1, i] = torch.std(Temp[n, i, :])

            fig, axis = plt.subplots(2, 1)
            nb = 7
            hist = torch.histc(D, bins=nb, min=0.0, max=6.0)
            axis[0].bar(range(nb), hist, align='center')
            axis[0].text(0, 100, torch.std(D).item())
            axis[0].set_title('DoW distribution')

            axis[1].plot(T[0, :])
            axis[1].set_title('T mean')
            # axis[2].plot(T[1, :])
            # axis[2].set_title('T std')

            plt.show()

    # Not used
    def search4ClosestCurves(self, X, loads, nCurves):
        # print("X: ", X)
        # print("loads ", loads)
        print('======start search4ClosestCurves()======')
        X_similar = torch.zeros([self.npoint, nCurves], dtype=torch.float32)
        scores = torch.zeros([nCurves])

        for c in range(nCurves):
            print('column No. ', c)
            min_dis = 1000000
            for n in range(len(X)):
                # print(' n: ', n)
                for j in range(self.nuser):
                    # print('     j: ', j)
                    dis = 0
                    for i in range(self.npoint):
                        r = X[n][i][j]
                        f = loads[0][i][c*2].item()
                        dis = dis + abs(r-f) * abs(r-f)
                        if dis > min_dis:
                            break
                    # dis = dis / self.npoint
                    # print('     dis: ', dis)

                    if dis < min_dis:
                        min_dis = dis
                        X_similar[:, c] = torch.tensor(X[n, :, j])

            scores[c] = min_dis

        fig, axis = plt.subplots(nCurves, 2)
        for c in range(nCurves):
            axis[c, 0].plot(X_similar[:, c], color='green')
            axis[c, 0].set_ylim((0, 5))

            axis[c, 1].plot(loads[0, :, c * 2])
            axis[c, 1].set_ylim((0, 5))
            axis[c, 1].text(0, 4, scores[c].item())
        # plt.ylim((0, 5))
        plt.show()

if __name__ == "__main__":
    E = Evaluator()

    # E.eval()
    # E.dispLoads()
    E.savesingledata()
    E.savegroupdata()
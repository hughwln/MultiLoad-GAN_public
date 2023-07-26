'''
Describe: Load and plot loss curves, and draw some example figures.
Author: Yi Hu
Email: yhu28@ncsu.edu
'''
import os
import glob

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from qqdm import qqdm
from torch.utils.tensorboard import SummaryWriter
import Data
import torch
import numpy as np
import kmeans_pytorch
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cm
import groupGAN
import SingleLoadGAN.singleUser_GAN as singleUser_GAN

import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
from sys import platform
import pandas as pd
import seaborn as sns
from scipy.spatial import distance

def smooth_curve(D, G, window_size = 20):
    smooth_D = []
    smooth_G = []
    for i in range(len(D)):
        start = max(0, i - window_size // 2)
        end = min(len(D), i + window_size // 2)
        value_D = np.mean(D[start: end])
        value_G = np.mean(G[start: end])
        smooth_D.append(value_D)
        smooth_G.append(value_G)
    return smooth_D, smooth_G

def load_loss_curve():
    if platform == "linux" or platform == "linux2":
        stage1_file = '/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/tensorboard/log/events.out.tfevents.1657201984.analytics01.1433568.0'
        stage2_file = '/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/tensorboard/logAuto/events.out.tfevents.1657217324.analytics01.1985658.0'
        singleGAN_file = '/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/singleUser/tensorboard/log/events.out.tfevents.1654525551.smarthome3.98936.0'
    elif platform == "darwin":
        stage1_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\tensorboard\\log\\events.out.tfevents.1657201984.analytics01.1433568.0'
        stage2_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\tensorboard\\logAuto\\events.out.tfevents.1657217324.analytics01.1985658.0'
        singleGAN_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\singleUser\\tensorboard\\log\\events.out.tfevents.1654525551.smarthome3.98936.0'
    elif platform == "win32":
        stage1_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\tensorboard\\log\\events.out.tfevents.1657201984.analytics01.1433568.0'
        stage2_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\tensorboard\\logAuto\\events.out.tfevents.1673624221.analytics02.40156.0'
        singleGAN_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\singleUser\\tensorboard\\log\\events.out.tfevents.1654525551.smarthome3.98936.0'
    G1 = []
    D1 = []
    G2 = []
    D2 = []

    for event in summary_iterator(stage1_file):
        for value in event.summary.value:
            # print(value)
            if value.tag == 'Loss/G':
                # print(value.tag)
                if value.HasField('simple_value'):
                    G1.append(value.simple_value)
            if value.tag == 'Loss/D':
                # print(value.tag)
                if value.HasField('simple_value'):
                    D1.append(value.simple_value)

    for event in summary_iterator(stage2_file):
        for value in event.summary.value:
            # print(value)
            if value.tag == 'Loss/G':
                # print(value.tag)
                if value.HasField('simple_value'):
                    G2.append(value.simple_value)
            if value.tag == 'Loss/D':
                # print(value.tag)
                if value.HasField('simple_value'):
                    D2.append(value.simple_value)

    # print('D1\n', D1)
    # print('G1\n', G1)
    # print('D2\n', D2)
    # print('G1\n', G2)

    D = D1 + D2
    G = G1 + G2
    smooth_D, smooth_G = smooth_curve(D, G)

    #     Single-GAN Loss
    G_single = []
    D_single = []
    for event in summary_iterator(singleGAN_file):
        for value in event.summary.value:
            # print(value)
            if value.tag == 'Loss/G':
                # print(value.tag)
                if value.HasField('simple_value'):
                    G_single.append(value.simple_value)
            if value.tag == 'Loss/D':
                # print(value.tag)
                if value.HasField('simple_value'):
                    D_single.append(value.simple_value)
    smooth_Gsingle, smooth_Dsingle = smooth_curve(D_single, G_single)

    fig, axis = plt.subplots(2, 1)
    plt.rc('font', size=12)  # controls default text sizes
    plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
    axis[0].plot(smooth_D, color='orange', label='Discriminator')
    axis[0].plot(smooth_G, color='green', label='Generator')
    axis[0].set_ylim([-4, 10])
    axis[0].set_xlabel('steps', loc='right')
    axis[0].set_ylabel('MultiLoad-GAN Loss', labelpad=2)
    axis[0].legend()

    axis[1].plot(smooth_Dsingle, color='orange', label='Discriminator')
    axis[1].plot(smooth_Gsingle, color='green', label='Generator')
    axis[1].set_ylim([-10, 20])
    axis[1].set_xlabel('steps', loc='right')
    axis[1].set_ylabel('SingleLoad-GAN Loss', labelpad=2)
    axis[1].legend()
    plt.subplots_adjust(hspace=0.3, top=0.95, bottom=0.11, left=0.1, right=0.95)
    plt.show()
    fig.savefig('/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/figures/loss_curve.svg', format="svg")

def colors():
    fig = plt.figure(0)
    plt.rc('font', size=12)  # controls default text sizes
    plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["green", "blue", "red", "black"])

    index = np.arange(0, 2*np.pi, 0.0001)
    data = 3*np.sin(index-np.pi/2)+3
    plt.scatter(index, data, c=data, marker='.', cmap=cmap, linewidths=0.1)
    plt.colorbar()
    line0 = np.full(len(index), 0)
    line3 = np.full(len(index), 2)
    line6 = np.full(len(index), 4)
    line9 = np.full(len(index), 6)
    plt.plot(index, line0, color='green', linestyle='--', linewidth=2)
    plt.plot(index, line3, color='blue', linestyle='--', linewidth=2)
    plt.plot(index, line6, color='red', linestyle='--', linewidth=2)
    plt.plot(index, line9, color='black', linestyle='--', linewidth=2)
    plt.xlabel('time', labelpad=0.5)
    plt.ylabel('load (kW)', labelpad=0.5)
    plt.subplots_adjust(top=0.95, bottom=0.11, left=0.11, right=0.95)
    plt.show()
    fig.savefig('/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/figures/encoding_examples.svg', format="svg")

def show_load_examples():
    if platform == "linux" or platform == "linux2":
        real_file = '/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/dataset/newRiver/GANData0.csv'
    elif platform == "darwin":
        real_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\GANData0.csv'
    elif platform == "win32":
        # real_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\GANData0.csv'
        real_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\groupGeneratedData_Auto.csv'
    real_data = pd.read_csv(real_file)

    real_loads = real_data.iloc[:, 1:-1].to_numpy()
    real_loads = real_loads.reshape((-1, 96, 8))

    for i in range(len(real_loads)):
        fig, axis = plt.subplots(8, 1)
        for c in range(8):
            label = range(96)
            axis[c % 8].plot(label, real_loads[i, :, c], color='blue')
            axis[c % 8].set_ylim((0, 2))
            axis[c % 8].set_xlabel('hour', loc='right', labelpad=1)
            axis[c % 8].set_yticks([])
            if c == 7:
                axis[c % 8].set_xticks(np.arange(0, 97, 4))
                axis[c % 8].set_xticklabels(np.arange(0, 25, 1))
            else:
                axis[c % 8].set_xticks([])
        bottom_margin = 0.11
        top_margin = 0.95
        left_margin = 0.12
        right_margin = 0.95
        plt.subplots_adjust(wspace=0, hspace=0, top=top_margin, bottom=bottom_margin, left=0.05, right=right_margin)
        plt.show()

    return real_loads

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def autocorr(x, lags):
    mean = np.mean(x)
    var = np.var(x)
    xp = x - mean
    corr = [1. if l==0 else np.sum(xp[l:]*xp[:-l])/len(x)/var for l in lags]

    return np.array(corr)

def eval_SingleLoad_gan():
    if platform == "linux" or platform == "linux2":
        workspace_dir = '/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/'
        ckpt_dir_single = os.path.join(workspace_dir, 'singleUser/checkpoints')
    elif platform == "darwin":
        workspace_dir = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public'
        ckpt_dir_single = os.path.join(workspace_dir, 'singleUser\\checkpoints')
    elif platform == "win32":
        workspace_dir = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public'
        ckpt_dir_single = os.path.join(workspace_dir, 'singleUser\\checkpoints')
    ckpt_dir = os.path.join(workspace_dir, 'checkpoints')

    z_dim = 100
    n_output = 16
    npoint = 96*7
    nuser = 8

    singleUser_G = singleUser_GAN.Generator(z_dim)
    singleUser_G.load_state_dict(torch.load(os.path.join(ckpt_dir_single, 'G_single_1.pth')))
    singleUser_G.eval()
    singleUser_G.cuda()

    times = 89 #89
    # real data
    real_data = Data.load_data()  # numpy  columns=8
    real_data = real_data.reshape((-1, 672, 8))
    real_data = np.random.permutation(real_data)[:16 * times]

    # single generation
    z_sample_single = Variable(torch.randn(n_output * nuser * times, z_dim)).cuda()
    loads_single = singleUser_G(z_sample_single).data
    # loads_single = loads_single * 5.0 + 5.0
    loads_single = loads_single * 3.0 + 3.0

    week_real = []
    for n in range(real_data.shape[0]):
        week = real_data[n]
        week_real.append(week)
    np_real = np.concatenate(week_real, axis=1)
    df_real = pd.DataFrame(np_real)

    groupdata = []
    for c in range(nuser*n_output*times):
        load = loads_single[c, 0].reshape((npoint, 1))
        groupdata.append(load)
    group = torch.cat(groupdata, dim=1).cpu()  # 1 day 16 users    96*16
    df_single = pd.DataFrame(group).astype(np.float32)

    # scatter plot of mean and std
    dis_mean_real = df_real.mean()
    dis_std_real = df_real.std()
    dis_mean_single = df_single.mean()
    dis_std_single = df_single.std()

    xlabel_distance = 1
    ylabel_distance = 4
    title_distance = 1.0
    fig_width = 6
    fig_height = 4
    bottom_margin = 0.11
    top_margin = 0.95
    left_margin = 0.12
    right_margin = 0.95
    plt.scatter(dis_mean_real.to_numpy(), dis_std_real.to_numpy(), c='r', label="Real", alpha=0.5, s=1)
    plt.scatter(dis_mean_single.to_numpy(), dis_std_single.to_numpy(), c='b', label="Generate", alpha=0.5, s=1)
    plt.ylabel('std', labelpad=ylabel_distance)
    plt.xlabel('kW', loc='right', labelpad=xlabel_distance)
    plt.legend(loc="upper right")
    plt.show()

    # Auto-correlation
    # lags = np.array(range(1, 200))
    # # df_real_3 = df_real.sample(3, axis=1)
    # # df_single_3 = df_single.sample(3, axis=1)
    # for i in range(len(df_real.columns)):
    #     fig, axis = plt.subplots(3, 1)
    #     corr_real = autocorr(df_real.iloc[:, i].to_numpy(), lags)
    #     corr_single = autocorr(df_single.iloc[:, i].to_numpy(), lags)
    #     axis[0].plot(df_single.iloc[:, i], c='blue')
    #     axis[1].plot(df_real.iloc[:, i], c='red')
    #     axis[2].plot(corr_single, c='blue')
    #     axis[2].plot(corr_real, c='red')
    #     plt.show()

    # clustering
    n_class = 5
    cluster_ids_x, cluster_centers = kmeans_pytorch.kmeans(
        X=torch.transpose(torch.tensor(np_real), 0, 1), num_clusters=n_class, distance='euclidean', device=torch.device('cuda:0')  # euclidean
    )
    group_ids = kmeans_pytorch.kmeans_predict(X=torch.transpose(group, 0, 1), cluster_centers=cluster_centers, device=torch.device('cuda:0'))
    js_cluster = distance.jensenshannon(cluster_ids_x, group_ids, 2.0)

    for i in range(n_class):
        plt.plot(cluster_centers[i])
    plt.show()

    for i in range(n_class):
        c_real_cluster = []
        for c in range(group_ids.size(dim=0)):
            if cluster_ids_x[c].item() == i:
                c_real_cluster.append(c)
        df_real_cluster = df_real[c_real_cluster]
        # mean curves
        mean_real = df_real_cluster.mean(axis=1)
        plt.plot(mean_real)
    plt.show()

    # for c in range(group_ids.size(dim=0)):
    #     n_cluster = group_ids[c].item()

    for i in range(n_class):
        print("===============Cluster # ", i)
        c_real_cluster = []
        c_single_cluster = []
        for c in range(group_ids.size(dim=0)):
            if cluster_ids_x[c].item() == i:
                c_real_cluster.append(c)
            if group_ids[c].item() == i:
                c_single_cluster.append(c)

        df_real_cluster = df_real[c_real_cluster]
        df_single_cluster = df_single[c_single_cluster]

        # mean curves
        mean_real = df_real_cluster.mean(axis=1)
        mean_single = df_single_cluster.mean(axis=1)
        mean_single, _ = smooth_curve(mean_single, [], 10)

        # RMSE
        err = rmse(mean_real.to_numpy(), mean_single)

        plt.plot(mean_real, 'r', label='Real')
        plt.plot(mean_single, 'b', label="Generate")
        plt.legend(loc="upper right")
        plt.show()

        # value distribution
        vec_real = df_real_cluster.to_numpy().reshape(-1, 1)
        vec_single = df_single_cluster.to_numpy().reshape(-1, 1)
        length = min(len(vec_real), len(vec_single))
        pdf_real, _, _ = plt.hist(vec_real[:length], bins=50, range=(0, 6), density=True)
        pdf_single, _, _ = plt.hist(vec_single[:length], bins=50, range=(0, 6), density=True)
        # J-S distance
        js = distance.jensenshannon(pdf_real, pdf_single, 10.0)
        print("rmse, ", err, " js, ", js)

        vec_real = np.random.permutation(vec_real)[:1000]
        vec_single = np.random.permutation(vec_single)[:1000]
        bhist = False

        nb = 50
        fig, axis = plt.subplots(1, 1, figsize=(fig_width, 3))
        sns.distplot(vec_real, bins=nb, hist=bhist, label='Real', color='red')
        sns.distplot(vec_single, bins=nb, hist=bhist, label='Generated', color='blue')
        plt.ylabel('Density', labelpad=ylabel_distance)
        plt.xlabel('kW', loc='right', labelpad=xlabel_distance)
        plt.yticks([])
        plt.legend(loc="upper right")
        plt.subplots_adjust(wspace=0.1, hspace=0, top=top_margin, bottom=0.15, left=left_margin, right=right_margin)
        plt.show()




if __name__ == '__main__':

    # load_loss_curve()
    # show_load_examples()
    # colors()
    eval_SingleLoad_gan()

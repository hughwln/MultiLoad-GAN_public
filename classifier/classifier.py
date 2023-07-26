'''
Describe: Build Classifier model and train it.
Author: Yi Hu
Email: yhu28@ncsu.edu
'''

import os
import glob
import numpy as np
import torch.nn as nn
import torch
from torch import autograd
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import matplotlib.pyplot as plt
from qqdm import qqdm
from torch.utils.tensorboard import SummaryWriter
from scipy.signal import find_peaks
import Data
from Data import NEWRIVERDataset
import pandas as pd
import random
from sys import platform

from collections import OrderedDict
from ignite.metrics import FID
from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *
import seaborn as sns

class Classifier(nn.Module):
    def __init__(self, dim=32, deep=False):
        super(Classifier, self).__init__()
        # The arguments for commonly used modules:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        # input image size: [3, 128, 128]
        # input image size: [1, 96*7, 8]
        if deep:
            self.cnn_layers = nn.Sequential(
                # 96*7*8
                nn.Conv2d(1, dim, (3, 3), (3, 1), (0, 1)),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
                nn.MaxPool2d(1, 1, 0),
                # 32*7*8
                nn.Conv2d(dim, 2 * dim, 3, 1, 1),
                nn.BatchNorm2d(2 * dim),
                nn.ReLU(),
                nn.MaxPool2d((2, 1), (2, 1), 0),
                # 16*7*8
                nn.Conv2d(2 * dim, 4 * dim, 3, 1, 1),
                nn.BatchNorm2d(4 * dim),
                nn.ReLU(),
                nn.MaxPool2d(2, 2, 0),
                # 8*7*4
                nn.Conv2d(4 * dim, 8 * dim, 3, 1, 1),
                nn.BatchNorm2d(8 * dim),
                nn.ReLU(),
                nn.MaxPool2d(2, 2, 0),
                # 4*7*2
                nn.Conv2d(8 * dim, 16 * dim, 3, 1, 1),
                nn.BatchNorm2d(16 * dim),
                nn.ReLU(),
                nn.MaxPool2d((4, 1), (4, 1), 0),
                # 7*2
            )
            self.fc_layers = nn.Sequential(
                nn.Linear(16 * dim * 7 * 2, 16 * dim),
                nn.ReLU(),
                nn.Linear(16 * dim, 8 * dim),
                nn.ReLU(),
                nn.Linear(8 * dim, 2 * dim),
                nn.ReLU(),
                nn.Linear(2 * dim, 16),
                nn.ReLU(),
                nn.Linear(16, 2),

                nn.Softmax(dim=-1)
            )
        else:
            self.cnn_layers = nn.Sequential(
                # 96*7*8
                nn.Conv2d(1, dim, (3, 3), (3, 1), (0, 1)),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
                nn.MaxPool2d(1, 1, 0),
                # 32*7*8
                nn.Conv2d(dim, 2*dim, 3, 1, 1),
                nn.BatchNorm2d(2*dim),
                nn.ReLU(),
                nn.MaxPool2d((4, 2), (4, 2), 0),
                # # 16*7*4
                # nn.Conv2d(2*dim, 4*dim, 3, 1, 1),
                # nn.BatchNorm2d(4*dim),
                # nn.ReLU(),
                # nn.MaxPool2d(2, 2, 0),
                # 8*7*4
                nn.Conv2d(2*dim, 4*dim, 3, 1, 1),
                nn.BatchNorm2d(4*dim),
                nn.ReLU(),
                nn.MaxPool2d(2, 2, 0),
                # 4*7*2
                nn.Conv2d(4*dim, 8*dim, 3, 1, 1),
                nn.BatchNorm2d(8*dim),
                nn.ReLU(),
                nn.MaxPool2d((4, 1), (4, 1), 0),
                # 7*2
            )
            self.fc_layers = nn.Sequential(
                nn.Linear(8*dim * 7 * 2, 8*dim),
                nn.ReLU(),
                nn.Linear(8*dim, dim),
                nn.ReLU(),
                nn.Linear(dim, 2),

                nn.Softmax(dim=-1)
            )

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]

        # Extract features by convolutional layers.
        x = self.cnn_layers(x)

        # The extracted feature map must be flatten before going to fully-connected layers.
        x = x.flatten(1)

        # The features are transformed by fully-connected layers to obtain the final logits.
        x = self.fc_layers(x)
        return x

def get_unlabel_data(X, model, ratio=2):
    device = 'cuda' if torch.cuda.is_available() else 'cup'
    # unsupervised samples
    model.eval()
    dfs = []
    for i in range(X.shape[0]):
        df = pd.DataFrame(X[i, 0])    # 672 * 8
        dfs.append(df)
    week_df = pd.concat(dfs, axis=1)

    fake_list = []
    for i in range(X.shape[0] // ratio):
        fake_group = week_df.sample(n=8, axis='columns')
        fake_group.columns = range(8)
        fake_group = fake_group.loc[:, fake_group.mean().sort_values(ascending=True).index]
        fake_group.columns = range(8)
        fake_list.append(fake_group)
    fake_data = pd.concat(fake_list)
    fake_data = fake_data.to_numpy(dtype=float)
    x = fake_data.reshape((-1, 1, 96 * 7, 8))
    y = []
    for i in range(x.shape[0]):
        input = torch.from_numpy(x[i:i+1]).float()
        output = model(input.to(device))
        y.append(output.argmax(dim=-1).item())

    dataset = NEWRIVERDataset(x, y)
    return dataset

def train(day = 7, shuffle='origin', buildUnlabelData=False):
    # "cuda" only when GPUs are available.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if day == 7:
        indim = 32
        d = True
    else:
        indim = 32
        d = True

    shuffleInTrain = True

    # Initialize a model, and put it on the device specified.
    model = Classifier(dim=indim, deep=d).to(device)
    model.device = device

    # For the classification task, we use cross-entropy as the measurement of performance.
    criterion = nn.CrossEntropyLoss()

    # Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # The number of training epochs.
    n_epochs = 1000
    batch_size = 64

    # load labeled training and testing set
    loader = Data.newRiverLoader()
    XY = loader.load_weekly_shifted_data_for_classifier(starting_day=day, shuffle=shuffle)
    XY = XY.iloc[:, 1:].to_numpy(dtype=float)
    XY = XY.reshape((-1, 96 * 7, 9))
    rand_indx = torch.randperm(XY.shape[0])
    XY = XY[rand_indx]
    X = XY[:, :, :-1]
    X = X.reshape((-1, 1, 96 * 7, 8))
    Y = XY[:, 0, -1]

    percent = int(X.shape[0] * (1 - 0.2))
    train_x, train_y, test_x, test_y = X[:percent], Y[:percent], X[percent:], Y[percent:]
    train_set = NEWRIVERDataset(train_x, train_y)
    test_set = NEWRIVERDataset(test_x, test_y)
    valid_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)

    #
    # trainDataFile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\loaderTrain' + str(
    #     day) + '.pt'
    # testDataFile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\loaderTest' + str(
    #     day) + '.pt'
    # train_loader = torch.load(trainDataFile)
    # valid_loader = torch.load(testDataFile)


    # ---------- Training ----------
    if shuffle == 'random':
        tb = SummaryWriter(f"tensorboard/log" + str(day))
    elif shuffle == 'sort':
        tb = SummaryWriter(f"tensorboard/logSorted" + str(day))
    elif shuffle == 'origin':
        tb = SummaryWriter(f"tensorboard/logOrigin" + str(day))
    steps = 0
    for e, epoch in enumerate(range(n_epochs)):
        if buildUnlabelData:
            print("building pseudo set...")
            unsupervised_set = get_unlabel_data(X, model, ratio=2)
            print("concat dataset")
            dataset = ConcatDataset([train_set, unsupervised_set])
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        else:
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        print("training...")
        model.train()
        train_progress_bar = qqdm(train_loader)

        # These are used to record information in training.
        train_losses = []
        train_accs = []

        # Iterate the training set by batches.
        for i, data in enumerate(train_progress_bar):
            # for batch in tqdm(train_loader):
            # A batch consists of image data and corresponding labels.
            imgs, labels = data
            imgs, labels = imgs.to(device), labels.to(device)
            # imgs = data[:, :, 0:8].to(device, dtype=torch.float) # RuntimeError: Input type (torch.cuda.DoubleTensor) and weight type (torch.cuda.FloatTensor) should be the same
            if shuffleInTrain:
                for idx in range(imgs.shape[0]):
                    rand_indx = torch.randperm(8)
                    # print(rand_indx)
                    imgs[idx] = imgs[idx, :, :, rand_indx]

            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(imgs)

            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = criterion(logits, labels)

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device, dtype=torch.float)).float().mean()

            # Record the loss and accuracy.
            train_losses.append(loss.item())
            train_accs.append(acc.item())

            # Print the information.
            steps += 1
            # print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
            train_progress_bar.set_infos({
                'Loss': round(loss.item(), 4),
                'acc': round(acc.item(), 4),
                'Epoch': epoch + 1,
                'Step': steps
            })

            tb.add_scalar("train/Loss", loss.item(), steps)
            tb.add_scalar("train/Acc", acc.item(), steps)

        # The average loss and accuracy of the training set is the average of the recorded values.
        train_loss = sum(train_losses) / len(train_losses)
        train_acc = sum(train_accs) / len(train_accs)

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()

        # These are used to record information in validation.
        eval_progress_bar = qqdm(valid_loader)

        # These are used to record information in training.
        valid_losses = []
        valid_accs = []

        # Iterate the training set by batches.
        for i, data in enumerate(eval_progress_bar):
            # A batch consists of image data and corresponding labels.
            imgs, labels = data
            imgs, labels = imgs.to(device), labels.to(device)
            # imgs = data[:, :, 0:8].to(device, dtype=torch.float) # RuntimeError: Input type (torch.cuda.DoubleTensor) and weight type (torch.cuda.FloatTensor) should be the same
            if shuffleInTrain:
                for idx in range(imgs.shape[0]):
                    rand_indx = torch.randperm(8)
                    imgs[idx] = imgs[idx, :, :, rand_indx]

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))

            # We can still compute the loss (but not the gradient).
            loss = criterion(logits, labels)

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device, dtype=torch.float)).float().mean()

            # Record the loss and accuracy.
            valid_losses.append(loss.item())
            valid_accs.append(acc.item())

            eval_progress_bar.set_infos({
                'eval_Loss': round(loss.item(), 4),
                'eval_acc': round(acc.item(), 4),
                'Epoch': epoch + 1,
            })

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_losses) / len(valid_losses)
        valid_acc = sum(valid_accs) / len(valid_accs)
        tb.add_scalar("eval/Loss", valid_loss, epoch+1)
        tb.add_scalar("eval/Acc", valid_acc, epoch+1)


        if shuffle == 'random':
            torch.save(model.state_dict(), './model/classifier'+str(day)+'.pth')
        elif shuffle == 'sort':
            torch.save(model.state_dict(), './model/classifierSorted' + str(day) + '.pth')
        elif shuffle == 'origin':
            torch.save(model.state_dict(), './model/classifierOrigin' + str(day) + '.pth')
        else:
            print('[Train] invalid shuffle!!!')

        if (epoch + 1) % 5 == 0 or epoch == 0:
            # acc_group = test_GAN(day=day, mode='group')
            # acc_single = test_GAN(day=day, mode='single')
            test_acc_real, confidence_real, acc_group, confidence_group, acc_single, confidence_single = test_GAN(day=day, ADA=False)
            tb.add_scalar("GAN_ACC/Group", acc_group, epoch + 1)
            tb.add_scalar("GAN_ACC/Single", acc_single, epoch + 1)

def test(model_day=7, data_day=0, shuffle='origin'):
    # ---------- Testing ----------
    # "cuda" only when GPUs are available.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_day == 7:
        indim = 32
        d = True
    else:
        indim = 32
        d = True
    # Initialize a model, and put it on the device specified.
    model = Classifier(dim=indim, deep=d).to(device)
    model.device = device
    if shuffle == 'random':
        model.load_state_dict(torch.load('./model/classifier'+str(model_day)+'.pth'))
    elif shuffle == 'sort':
        model.load_state_dict(torch.load('/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/classifier/model/classifierSorted' + str(model_day) + '_Impressive.pth'))
        # model.load_state_dict(torch.load('/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/classifier/model/classifierAuto.pth'))
    elif shuffle == 'origin':
        model.load_state_dict(torch.load('./model/classifierOrigin' + str(model_day) + '.pth'))
    else:
        print('[Test] invalid shuffle!!!')
    model.eval()

    if platform == "linux" or platform == "linux2":
        # testDataFile = '~/Documents/Code/Research/MultiLoad-GAN_public/dataset/newRiver/loaderTest' + str(
        # data_day) + '.pt'
        testDataFile = '/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/dataset/newRiver/loaderTest0.pt'
    elif platform == "darwin":
        testDataFile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\loaderTest' + str(
        data_day) + '.pt'
    elif platform == "win32":
        testDataFile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\loaderTest' + str(
        data_day) + '.pt'

    test_loader = torch.load(testDataFile)

    progress_bar = qqdm(test_loader)

    # These are used to record information in training.
    test_accs = []
    shuffleInTest = False

    # Iterate the training set by batches.
    for i, data in enumerate(progress_bar):
        # A batch consists of image data and corresponding labels.
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)
        if shuffleInTest:
            for idx in range(imgs.shape[0]):
                rand_indx = torch.randperm(8)
                imgs[idx] = imgs[idx, :, rand_indx]

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device, dtype=torch.float)).float().mean()
        # Record the loss and accuracy.
        test_accs.append(acc.item())

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    test_acc = sum(test_accs) / len(test_accs)
    print(test_acc)

    return test_acc

def test_GAN(day=7, ADA=False):
    # ---------- Testing ----------
    # "cuda" only when GPUs are available.
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    # Initialize a model, and put it on the device specified.
    model = Classifier(dim=32, deep=True).to(device)
    model.device = device
    # model.load_state_dict(torch.load('/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/classifier/model/classifierSorted' + str(day) + '_paper.pth'))         # final model
    model.load_state_dict(torch.load( 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\classifier\\model\\classifierSorted' + str(day) + '_paper.pth'))
    # model.load_state_dict(torch.load('/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/classifier/model/classifierSorted' + str(day) + '_Impressive.pth'))  # updated by data augmentation
    # model.load_state_dict(torch.load('/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/classifier/model/classifierAuto_enc.pth'))
    model.eval()

    if platform == "linux" or platform == "linux2":
    # linux
        lstm_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\lstmGeneratedData.csv'
        single_file = '/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/dataset/newRiver/groupGeneratedData_enc_para.csv'
        if ADA==False:
            group_file = '/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/dataset/newRiver/groupGeneratedData_Impressive.csv'        # W/O data augmentation
        else:
            group_file = '/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/dataset/newRiver/groupGeneratedData_Auto_enc.csv'            # with data augmentation
        real_file = '/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/dataset/newRiver/GANData0.csv'
    elif platform == "darwin":
    # OS X
        lstm_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\lstmGeneratedData.csv'
        single_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\singleGeneratedData.csv'
        group_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\groupGeneratedData.csv'
        real_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\GANData0.csv'
    elif platform == "win32":
    # Windows...
        lstm_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\lstmGeneratedData.csv'
        single_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\singleGeneratedData.csv'
        # single_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\groupGeneratedData_enc_para.csv'
        if ADA == False:
            group_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\groupGeneratedData_Impressive.csv'
        else:
            group_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\groupGeneratedData_Auto.csv'
        real_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\GANData0.csv'

    data_real = pd.read_csv(real_file)
    data_group = pd.read_csv(group_file)
    data_single = pd.read_csv(single_file)
    data_lstm = pd.read_csv(lstm_file)

    X_real = data_real.iloc[:, 1:-1].to_numpy()
    X_real = X_real.reshape((-1, 1, 96 * 7, 8))
    # plt.plot(X_real[0,0,:,-1])
    # plt.show()
    input_real = torch.tensor(X_real).to(device, dtype=torch.float)
    # shuffle input data
    for idx in range(input_real.shape[0]):
        rand_indx = torch.randperm(8)
        input_real[idx] = input_real[idx, :, :, rand_indx]
    output_real = model(input_real)
    labels_real = torch.ones(input_real.shape[0])

    X_group = data_group.iloc[:, 1:].to_numpy()
    X_group = X_group.reshape((-1, 1, 96 * 7, 8))
    input_group = torch.tensor(X_group).to(device, dtype=torch.float)
    # shuffle input data
    for idx in range(input_group.shape[0]):
        rand_indx = torch.randperm(8)
        input_group[idx] = input_group[idx, :, :, rand_indx]
    output_group = model(input_group)
    labels_group = torch.ones(input_group.shape[0])

    X_single = data_single.iloc[:, 1:].to_numpy()
    X_single = X_single.reshape((-1, 1, 96 * 7, 8))
    input_single = torch.tensor(X_single).to(device, dtype=torch.float)
    # shuffle input data
    for idx in range(input_single.shape[0]):
        rand_indx = torch.randperm(8)
        input_single[idx] = input_single[idx, :, :, rand_indx]
    output_single = model(input_single)
    labels_single = torch.ones(input_single.shape[0])

    X_lstm = data_lstm.iloc[:, 1:].to_numpy()
    X_lstm = X_lstm.reshape((-1, 1, 96 * 7, 8))
    input_lstm = torch.tensor(X_lstm).to(device, dtype=torch.float)
    # shuffle input data
    for idx in range(input_lstm.shape[0]):
        rand_indx = torch.randperm(8)
        input_lstm[idx] = input_lstm[idx, :, :, rand_indx]
    output_lstm = model(input_lstm)
    labels_lstm = torch.ones(input_lstm.shape[0])

    # Compute the accuracy for current batch.
    test_acc_real = (output_real.argmax(dim=-1) == labels_real.to(device, dtype=torch.float)).float().mean()
    print('real acc: ', test_acc_real.item())
    confidence_tensor_real = output_real[:, 1]
    confidence_real = confidence_tensor_real.reshape(-1).detach().numpy()

    test_acc_group = (output_group.argmax(dim=-1) == labels_group.to(device, dtype=torch.float)).float().mean()
    print('group acc: ', test_acc_group.item())
    confidence_tensor_group = output_group[:, 1]
    confidence_group = confidence_tensor_group.reshape(-1).detach().numpy()

    test_acc_single = (output_single.argmax(dim=-1) == labels_single.to(device, dtype=torch.float)).float().mean()
    print('single acc: ', test_acc_single.item())
    confidence_tensor_single = output_single[:, 1]
    confidence_single = confidence_tensor_single.reshape(-1).detach().numpy()

    test_acc_lstm = (output_lstm.argmax(dim=-1) == labels_lstm.to(device, dtype=torch.float)).float().mean()
    print('lstm acc: ', test_acc_lstm.item())
    confidence_tensor_lstm = output_lstm[:, 1]
    confidence_lstm = confidence_tensor_lstm.reshape(-1).detach().numpy()

    # ===========spot the outputs=================
    # plt.scatter(output_real[:, 0].reshape(-1).detach().numpy(), confidence_real, c='green')
    # plt.scatter(output_group[:, 0].reshape(-1).detach().numpy(), confidence_group, c='blue')
    # plt.scatter(output_single[:, 0].reshape(-1).detach().numpy(), confidence_single, c='red')
    # # plt.hist(confidence_group, bins=nb, range=(confidence_real.min(), confidence_real.max()), density=True,
    # #          histtype='step', label='group', color='blue')
    # # plt.hist(confidence_single, bins=nb, range=(confidence_real.min(), confidence_real.max()), density=True,
    # #          histtype='step', label='single', color='red')
    # plt.legend()
    # plt.title("Confidence Distribution")
    # plt.show()
    # ============================================

    return test_acc_real, confidence_real, test_acc_group, confidence_group, test_acc_single, confidence_single, test_acc_lstm, confidence_lstm

# input torch.tensor
def test_GAN_from_data(loads, mode='group'):
    # "cuda" only when GPUs are available.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize a model, and put it on the device specified.
    model = Classifier(dim=32, deep=True).to(device)
    model.device = device
    if platform == "linux" or platform == "linux2":
    # linux
    #     model_path = './classifier/model/classifierAuto.pth'
        model_path = '/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/classifier/model/classifierSorted0_Impressive.pth'
    elif platform == "darwin":
    # OS X
        model_path = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\classifier\\model\\classifierSorted0_Impressive.pth'
    elif platform == "win32":
    # Windows...
        model_path = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\classifier\\model\\classifierSorted0_Impressive.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()

    X = loads.reshape((-1, 1, 96 * 7, 8))
    input_data = X.to(device, dtype=torch.float)
    # shuffle input data
    for idx in range(input_data.shape[0]):
        rand_indx = torch.randperm(8)
        input_data[idx] = input_data[idx, :, :, rand_indx]

    output_data = model(input_data)
    if mode == 'group':
        labels = torch.ones(input_data.shape[0])
    elif mode == 'single':
        labels = torch.ones(input_data.shape[0])

    # Compute the accuracy for current batch.
    test_acc = (output_data.argmax(dim=-1) == labels.to(device, dtype=torch.float)).float().mean()
    print('test acc: ', test_acc.item())
    return test_acc


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


def confidence_distribution_distance():
    test_acc_real, confidence_real, test_acc_group, confidence_group, test_acc_single, confidence_single, test_acc_lstm, confidence_lstm = test_GAN(
        day=0, ADA=True)

    confidence_real = confidence_real[:1280]
    confidence_group = np.repeat(confidence_group, 4)
    confidence_single = np.repeat(confidence_single, 4)
    confidence_lstm = np.repeat(confidence_lstm, 2)
    confidence_lstm = confidence_lstm[:1280]

    fid_group = calculate_fid(confidence_real, confidence_group)
    fid_single = calculate_fid(confidence_real, confidence_single)
    fid_lstm = calculate_fid(confidence_real, confidence_lstm)
    print("FID group: ", fid_group)
    print("FID single: ", fid_single)
    print("FID lstm: ", fid_lstm)
    # metric = FID(num_features=1, feature_extractor=default_model)
    # metric.attach(default_evaluator, "fid")
    # # state = default_evaluator.run([[torch.from_numpy(confidence_real).reshape((-1, 1)), torch.from_numpy(confidence_group).reshape((-1, 1))]])
    # # print("FID group: ", state.metrics["fid"])
    # state = default_evaluator.run(
    #     [[torch.from_numpy(confidence_real).reshape((-1, 1)), torch.from_numpy(confidence_single).reshape((-1, 1))]])
    # print("FID single: ", state.metrics["fid"])
    # =====================================================================

    print("real_confidence", np.mean(confidence_real))
    print("group_confidence", np.mean(confidence_group))
    print("single_confidence", np.mean(confidence_single))
    print("lstm_confidence", np.mean(confidence_lstm))
    nb = 10
    fig = plt.figure(figsize=(2, 2.5))
    plt.rc('font', size=10)  # controls default text sizes
    plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
    plt.hist(confidence_real, bins=nb, density=True, histtype='step', label='REAL', color='green')
    plt.hist(confidence_group, bins=nb, range=(confidence_real.min(), confidence_real.max()), density=True,
             histtype='step', label='MLGAN', color='blue')
    plt.hist(confidence_single, bins=nb, range=(confidence_real.min(), confidence_real.max()), density=True,
             histtype='step', label='SLGAN', color='orange')
    plt.hist(confidence_lstm, bins=nb, range=(confidence_real.min(), confidence_real.max()), density=True,
             histtype='step', label='LSTM', color='red')
    plt.xlabel('score', labelpad=0.5)
    plt.ylabel('Density', labelpad=0.5)
    plt.yticks([])
    plt.legend(loc='upper left')
    # plt.title("Confidence Distribution")
    plt.subplots_adjust(top=0.95, bottom=0.15, left=0.1, right=0.96)
    plt.show()
    # fig.savefig('/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public/figures/cdd_ADA.svg', format="svg")

if __name__ == '__main__':
    # day == 7 shift model
    # day == 0-6 fixed start/end time model
    # for i in range(1, 8):
    #     print('Training model No. ', i)
    #     train(day=i, shuffle='sort')
    train(day=0, shuffle='sort', buildUnlabelData=False)
    #
    # accs = np.zeros([8, 8])
    # for i in range(8):
    #     for j in range(8):
    #         accs[i][j] = test(model_day=i, data_day=j, shuffle='sort')
    # print('accs: \n', accs)
    # test(model_day=0, data_day=0, shuffle='sort')
    #
    # group_result = []
    # single_result = []
    # for day in range(1):
    #     accs_group = []
    #     accs_single = []
    #     for i in range(10):
    #         acc_group = test_GAN(day=day, mode='group')
    #         accs_group.append(acc_group)
    #         acc_single = test_GAN(day=day, mode='single')
    #         accs_single.append(acc_single)
    #     group_result.append(np.mean(accs_group))
    #     single_result.append(np.mean(accs_single))
    # print(group_result)
    # print(single_result)

    confidence_distribution_distance()


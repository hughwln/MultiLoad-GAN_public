'''
Describe: Training in the Automatic Data Augmentation stage.
Author: Yi Hu
Email: yhu28@ncsu.edu
'''
import random
import torch
import numpy as np
from sys import platform

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# same_seeds(2021)

import os
import glob

import torch.nn as nn
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
import groupGAN
import classifier.classifier as classifier
import newRiver

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # same_seeds(2021)
    npoint = 96*7
    nuser = 8

    # ======================================
    # GAN model setup
    # ======================================
    batch_size_GAN = 16
    z_dim = 100
    z_sample = Variable(torch.randn(100, z_dim)).to(device)
    lr = 1e-4
    clip_value = 0.01
    w_gp = 10

    if platform == "linux" or platform == "linux2":
        workspace_dir = '/home/yhu28/Documents/Code/Research/MultiLoad-GAN_public'
    elif platform == "darwin":
        workspace_dir = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public'
    elif platform == "win32":
        workspace_dir = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public'
    log_dir = os.path.join(workspace_dir, 'logs')
    ckpt_dir = os.path.join(workspace_dir, 'checkpoints')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Model
    G = groupGAN.Generator(in_dim=z_dim, dim=80).to(device)
    D = groupGAN.Discriminator(in_dim=4, dim=32).to(device)
    # G.load_state_dict(torch.load('./checkpoints/G_Impressive.pth'))
    # D.load_state_dict(torch.load('./checkpoints/D_Impressive.pth'))
    G.load_state_dict(torch.load('./checkpoints/G.pth'))
    D.load_state_dict(torch.load('./checkpoints/D.pth'))
    G.train()
    D.train()

    # Loss
    criterion_GAN = nn.BCELoss()

    """ Medium: Use RMSprop for WGAN. """
    # Optimizer
    # opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    # opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.RMSprop(D.parameters(), lr=lr)
    opt_G = torch.optim.RMSprop(G.parameters(), lr=lr*1.4)

    # for GAN data augmentation
    NR = newRiver.newRiver()
    NR.load_data()

    loader_GAN = Data.newRiverLoader()
    trainData_GAN = loader_GAN.load_data_for_groupGAN(day=0)

    # ======================================
    # Classifier model setup
    # ======================================
    indim = 32
    d = True
    shuffleInTrain = True
    buildUnlabelData = True

    model = classifier.Classifier(dim=indim, deep=d).to(device)
    model.load_state_dict(torch.load('./classifier/model/classifierSorted0_Impressive.pth'))
    model.device = device

    # For the classification task, we use cross-entropy as the measurement of performance.
    criterion_classifier = nn.CrossEntropyLoss()

    # Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    opt_C = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # The number of training epochs.
    batch_size_C = 64

    # load labeled training and testing set
    loader_C = Data.newRiverLoader()
    XY = loader_C.load_weekly_shifted_data_for_classifier(starting_day=0, shuffle='sort')
    XY = XY.iloc[:, 1:].to_numpy(dtype=float)
    XY = XY.reshape((-1, 96 * 7, 9))
    rand_indx = torch.randperm(XY.shape[0])
    XY = XY[rand_indx]
    X = XY[:, :, :-1]
    X = X.reshape((-1, 1, 96 * 7, 8))
    Y = XY[:, 0, -1]

    percent = int(X.shape[0] * (1 - 0.2))
    train_x, train_y, test_x, test_y = X[:percent], Y[:percent], X[percent:], Y[percent:]
    train_set = Data.NEWRIVERDataset(train_x, train_y)
    test_set = Data.NEWRIVERDataset(test_x, test_y)
    valid_loader = DataLoader(test_set, batch_size=batch_size_C, shuffle=True, num_workers=2)

    # ===========================
    # Training loop
    # ===========================
    tb = SummaryWriter(f"tensorboard/logAuto_enc")
    steps = 0
    steps_GAN = 0
    n_epoch = 50  # 50
    n_critic = 1  # 5
    ratio = 1
    for e, epoch in enumerate(range(n_epoch)):
        # ==============build training set for Classifier=================
        # z = Variable(torch.randn(5000, z_dim)).to(device)
        # zs = []
        # for i in range(z.shape[0]):
        #     ff = (G(z[i:i+1]).data + 1.0) / 2.0
        #     zs.append(ff)
        # f_imgs = torch.cat(zs)

        times = 2
        ygroup = []
        G.eval()
        model.eval()
        for i in range(times):
            z_sample_single = Variable(torch.randn(16, z_dim)).to(device)
            imgs_group = (G(z_sample_single).data + 1.0) / 2.0
            ygroup.append(imgs_group)
        f_imgs = torch.cat(ygroup)

        # add figure to tensorboard
        generated_tensor, temperature = Data.decode_16users(f_imgs, npoint, nuser)
        generated_loads = generated_tensor.numpy()
        x_gen = generated_loads.reshape((-1, 1, 96 * 7, 8))
        y_gen = []
        for i in range(x_gen.shape[0]):
            input = torch.from_numpy(x_gen[i:i + 1]).float()
            output = model(input.to(device))
            y_gen.append(output.argmax(dim=-1).item())

        gen_set = Data.NEWRIVERDataset(x_gen, y_gen)

        if buildUnlabelData:
            print("building pseudo set...")
            unsupervised_set = classifier.get_unlabel_data(X, model, ratio=2)
            print("concat dataset")
            dataset = ConcatDataset([train_set, unsupervised_set, gen_set])
            train_loader = DataLoader(dataset, batch_size=batch_size_C, shuffle=True, num_workers=2, drop_last=True)
        else:
            train_loader = DataLoader(train_set, batch_size=batch_size_C, shuffle=True, num_workers=2, drop_last=True)
        print("training Classifier...")
        # ================================================================

        # ==============train Classifier==================================
        model.train()
        progress_bar_C = qqdm(train_loader)
        train_losses = []
        train_accs = []

        for i, data in enumerate(progress_bar_C):
            imgs, labels = data
            imgs, labels = imgs.to(device), labels.to(device)
            if shuffleInTrain:
                for idx in range(imgs.shape[0]):
                    rand_indx = torch.randperm(8)
                    imgs[idx] = imgs[idx, :, :, rand_indx]

            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(imgs)
            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = criterion_classifier(logits, labels)
            # Gradients stored in the parameters in the previous step should be cleared out first.
            opt_C.zero_grad()
            loss.backward()
            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            opt_C.step()
            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device, dtype=torch.float)).float().mean()
            # Record the loss and accuracy.
            train_losses.append(loss.item())
            train_accs.append(acc.item())

            steps += 1
            progress_bar_C.set_infos({
                'Loss': round(loss.item(), 4),
                'acc': round(acc.item(), 4),
                'Epoch': epoch + 1,
                'Step': steps
            })

            tb.add_scalar("train/Loss", loss.item(), steps)
            tb.add_scalar("train/Acc", acc.item(), steps)
        # ================================================================

        # =================evaluate Classifier============================
        model.eval()
        eval_progress_bar = qqdm(valid_loader)

        valid_losses = []
        valid_accs = []

        for i, data in enumerate(eval_progress_bar):
            imgs, labels = data
            imgs, labels = imgs.to(device), labels.to(device)
            if shuffleInTrain:
                for idx in range(imgs.shape[0]):
                    rand_indx = torch.randperm(8)
                    imgs[idx] = imgs[idx, :, :, rand_indx]

            with torch.no_grad():
                logits = model(imgs.to(device))

            loss = criterion_classifier(logits, labels)
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
        tb.add_scalar("eval/Loss", valid_loss, epoch + 1)
        tb.add_scalar("eval/Acc", valid_acc, epoch + 1)

        torch.save(model.state_dict(), './classifier/model/classifierAuto_enc.pth')

        # if (epoch + 1) % ratio == 0 or epoch == 0:
        #     acc_group = classifier.test_GAN_from_data(loads=generated_tensor, mode='group')
        #     acc_single = classifier.test_GAN(day=0, mode='single')
        #     tb.add_scalar("GAN_ACC/Group", acc_group, epoch + 1)
        #     tb.add_scalar("GAN_ACC/Single", acc_single, epoch + 1)
        # ================================================================

        # ========train GAN================
        if (e + 1) % ratio == 0:
            # ==============build training set for GAN=================
            model.eval()
            aug_data = NR.get_random_seleted_group(times=2)
            aug_data = aug_data.reshape((-1, 1, 672, 9))
            # aug_data = ConcatDataset([unsupervised_set, gen_set])
            aug_loader = DataLoader(aug_data, batch_size=4, shuffle=False, num_workers=2)
            temp_idx = []
            for i, data in enumerate(aug_loader):
                aug_x9 = data
                aug_x8 = aug_x9[:, :, :, :-1]
                for idx in range(aug_x8.shape[0]):
                    rand_indx = torch.randperm(8)
                    aug_x8[idx] = aug_x8[idx, :, :, rand_indx]
                output = model(aug_x8.to(device, dtype=torch.float))
                for j in range(4):
                    if output[j, 1].item() > 0.9:
                        temp_idx.append(aug_x9[j])
            if len(temp_idx):
                aug_set = torch.cat(temp_idx)    # N*672*9 tensor
                aug_numpy = aug_set.numpy()
                # idex = np.random.choice(range(len(aug_numpy)), size=500, replace=False)
                # aug_numpy = aug_numpy[idex]
                print("Encoding augmented data for GAN...", aug_numpy.shape[0])
                # torch.tensor
                aug_set_tensor = Data.encode_users(aug_numpy, npoint=672, nuser=8)
                aug_set_tensor = aug_set_tensor * 2.0 - 1.0

                trainset_GAN = torch.cat([aug_set_tensor, trainData_GAN])
                dataloader_GAN = DataLoader(trainset_GAN, batch_size=batch_size_GAN, shuffle=True, num_workers=2, drop_last=True)
            else:
                print("No augmented data for GAN")
                dataloader_GAN = DataLoader(trainData_GAN, batch_size=batch_size_GAN, shuffle=True, num_workers=2)
            # ================================================================

            print('training GAN ...')
            progress_bar_GAN = qqdm(dataloader_GAN)
            G.train()
            for i, data in enumerate(progress_bar_GAN):
                imgs = data.to(device, dtype=torch.float)
                bs = imgs.size(0)
                # get real image grid
                if i == 0:
                    images = imgs[0:6, 0:3]
                    images = (images + 1.0) / 2.0
                    r_grid_img = torchvision.utils.make_grid(images, nrow=6)
                    print("get real images grid")

                # ============================================
                #  Train D
                #  Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
                # ============================================
                z = Variable(torch.randn(bs, z_dim)).to(device, dtype=torch.float)
                r_imgs = Variable(imgs).to(device, dtype=torch.float)
                f_imgs = G(z)

                # WGAN Loss
                loss_D = -torch.mean(D(r_imgs)) + torch.mean(D(f_imgs))
                # compute Gradient Penalty
                gradient_penalty = groupGAN.compute_gp(D, r_imgs, f_imgs)
                loss_D = loss_D + w_gp * gradient_penalty
                # Model backwarding
                D.zero_grad()
                loss_D.backward()
                # Update the discriminator.
                opt_D.step()

                # ============================================
                #  Train G
                #  Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
                # ============================================
                if steps_GAN % n_critic == 0:
                    z = Variable(torch.randn(bs, z_dim)).to(device, dtype=torch.float)
                    f_imgs = G(z)
                    # WGAN Loss
                    loss_G = -torch.mean(D(f_imgs))
                    # Model backwarding
                    G.zero_grad()
                    loss_G.backward()
                    # Update the generator.
                    opt_G.step()

                steps_GAN += 1

                # Set the info of the progress bar
                progress_bar_GAN.set_infos({
                    'Loss_D': round(loss_D.item(), 4),
                    'Loss_G': round(loss_G.item(), 4),
                    'Epoch': e + 1,
                    'Step': steps_GAN
                })
                tb.add_scalar("Loss/G", loss_G, steps_GAN)
                tb.add_scalar("Loss/D", loss_D, steps_GAN)



        # =================evaluate GAN ==================================
        if (e + 1) % ratio == 0:
            G.eval()
            f_imgs_sample = (G(z_sample[:8]).data + 1.0) / 2.0

            # show generated load curves
            loads, temperature = Data.decode_16users(f_imgs_sample, npoint, nuser)
            fig, axis = plt.subplots(8, 1)
            for i in range(loads.shape[2]):
                axis[i % 8].plot(loads[0, :, i])
            tb.add_figure('profiles', fig, e+1)

            # f_imgs_sample = G(z_sample).data
            f_imgs_T = f_imgs_sample[:, 3]
            f_imgs_T = f_imgs_T.reshape((f_imgs_T.shape[0], 1, npoint, nuser))
            f_imgs_sample = f_imgs_sample[:, 0:3]

            # Show generated images in
            grid_img = torchvision.utils.make_grid(f_imgs_sample[0:6].cpu(), nrow=6)
            tb.add_image("images/fake", grid_img, e + 1)
            tb.add_image("images/real", r_grid_img, e + 1)
            grid_T = torchvision.utils.make_grid(f_imgs_T[0:6].cpu(), nrow=6)
            tb.add_image("Other/T", grid_T, e + 1)

            # Save the checkpoints.
            torch.save(G.state_dict(), os.path.join(ckpt_dir, 'G_Auto_enc.pth'))
            torch.save(D.state_dict(), os.path.join(ckpt_dir, 'D_Auto_enc.pth'))
        # ==================================================================

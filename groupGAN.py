'''
Describe: Build and Train MultiLoad-GAN
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
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from qqdm import qqdm
from torch.utils.tensorboard import SummaryWriter
from scipy.signal import find_peaks
import Data
# import GAN
import classifier.classifier as classifier

# ==========================
# GAN Model
# =============================

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    """
    Input shape: (N, in_dim)
    Output shape: (N, 4, 672, 8)
    """

    def __init__(self, in_dim, dim=64):     # 100, 80
        super(Generator, self).__init__()

        def dconv_bn_relu(in_dim, out_dim, k, s, p, output_p):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, k, s,
                                   padding=p, output_padding=output_p, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )

        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU()
        )
        # 5 2 2 1           *2
        # 5 1 2 0           *1
        # 3 3 0 0           *3
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 8, (7, 5), (7, 1), (0, 2), (0, 0)),  # 7*4*4
            dconv_bn_relu(dim * 8, dim * 4, 5, (2, 1), 2, (1, 0)),    # 7*8*4
            dconv_bn_relu(dim * 4, dim * 2, 5, (2, 1), 2, (1, 0)),    # 7*16*4
            dconv_bn_relu(dim * 2, dim, 5, 2, 2, 1),    # 7*32*8

            nn.ConvTranspose2d(dim, 4, (3, 5), (3, 1), padding=(0, 2), output_padding=(0, 0)),  # 7*96*8
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y


class Discriminator(nn.Module):
    """
    Input shape: (N, 4, 96, 8)
    Output shape: (N, )
    """

    def __init__(self, in_dim, dim=64):     # 4, 32
        super(Discriminator, self).__init__()

        def conv_bn_lrelu(in_dim, out_dim, k, s, p=0):
            return nn.Sequential(
                # nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                nn.Conv2d(in_dim, out_dim, k, s, p),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2),
            )

        """ Medium: Remove the last sigmoid layer for WGAN. """
        # 5 2 2           *2
        # 5 1 2           *1
        # 3 3 0           *3
        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, (7, 5), (7, 1), (0, 2)),     # 96*8
            nn.LeakyReLU(0.2),

            conv_bn_lrelu(dim, dim*2, (3, 5), (3, 1), (0, 2)),        # 32*8
            conv_bn_lrelu(dim*2, dim * 4, 5, (2, 1), 2),    # 16*4
            # conv_bn_lrelu(dim * 4, dim * 4, 5, (2, 1), 2),    # 8*4
            conv_bn_lrelu(dim * 4, dim * 8, (4 , 5), (4, 1), (0, 2)),         # 4*4
            nn.Conv2d(dim * 8, 1, 4),           # 1*1
            # nn.Sigmoid(),
        )
        self.apply(weights_init)

    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y

# ==================================
# compute gradient penalty
# ==================================
def compute_gp(netD, real_data, fake_data):
    batch_size = real_data.size(0)
    # Sample Epsilon from uniform distribution
    eps = torch.rand(batch_size, 1, 1, 1).to(real_data.device)
    eps = eps.expand_as(real_data)

    # Interpolation between real data and fake data.
    interpolation = eps * real_data + (1 - eps) * fake_data

    # get logits for interpolated images
    interp_logits = netD(interpolation)
    grad_outputs = torch.ones_like(interp_logits)

    # Compute Gradients
    gradients = autograd.grad(
        outputs=interp_logits,
        inputs=interpolation,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]

    # Compute and return Gradient Norm
    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, 1)
    return torch.mean((grad_norm - 1) ** 2)

# use 0 2 4 6 8 10 12 14
def search4ClosestCurves(X, loads, nCurves):
    # print("X: ", X)
    # print("loads ", loads)
    # print('======start search4ClosestCurves()======')
    X_similar = torch.zeros([npoint, nCurves], dtype=torch.float32)
    scores = torch.zeros([nCurves])

    for c in range(nCurves):
        print('column No. ', c)
        min_dis = 1000000
        for n in range(len(X)):
            # print(' n: ', n)
            for j in range(nuser):
                # print('     j: ', j)
                dis = 0
                for i in range(npoint):
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
    return X_similar, scores

if __name__ == '__main__':
    same_seeds(2021)
    npoint = 96*7
    nuser = 8
    # X = Data.load_data()  # numpy  columns=16 + 2
    # X = X.reshape((-1, npoint, nuser+2))
    # X = Data.encode_16users(X, npoint, nuser)

    # ======================================
    # Training
    # ======================================
    # Training hyperparameters
    batch_size = 16
    z_dim = 100
    z_sample = Variable(torch.randn(100, z_dim)).cuda()
    # z_sample1 = Variable(torch.randn(100, z_dim))
    # plt.plot(z_sample1[0])
    # plt.show()
    lr = 1e-4
    # lr = 5e-5

    """ Medium: WGAN, 50 epoch, n_critic=5, clip_value=0.01 """
    n_epoch = 300  # 50
    n_critic = 1  # 5
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
    G = Generator(in_dim=z_dim, dim=80).cuda()
    D = Discriminator(in_dim=4, dim=32).cuda()
    G.train()
    D.train()

    # Loss
    criterion = nn.BCELoss()

    """ Medium: Use RMSprop for WGAN. """
    # Optimizer
    # opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    # opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.RMSprop(D.parameters(), lr=lr)
    opt_G = torch.optim.RMSprop(G.parameters(), lr=lr*1.4)

    # DataLoader
    # dataloader = DataLoader(X, batch_size=batch_size, shuffle=True, num_workers=2)
    loader = Data.newRiverLoader()
    trainData = loader.load_data_for_groupGAN(day=0)
    print('Creating dataloader...')
    dataloader = DataLoader(trainData, batch_size=batch_size, shuffle=True, num_workers=2)
    print('Done!')

    tb = SummaryWriter(f"tensorboard/log")

    # ===========================
    # Training loop
    # ===========================

    steps = 0
    for e, epoch in enumerate(range(n_epoch)):
        progress_bar = qqdm(dataloader)
        G.train()
        for i, data in enumerate(progress_bar):

            # imgs = Data.encode_users(data, npoint=npoint, nuser=nuser)
            # transform from [0, 1] to [-1, 1]
            # imgs = imgs * 2.0 - 1.0
            # print('data\n', imgs)
            imgs = data.cuda()
            bs = imgs.size(0)
            # print('bs: ', imgs.shape)

            # get real image grid
            if i == 0:
                images = imgs[0:6, 0:3]
                # print(images)
                # images = images[:, 0:3]
                images = (images + 1.0) / 2.0
                r_grid_img = torchvision.utils.make_grid(images, nrow=6)
                print("get real images grid")

            # ============================================
            #  Train D
            #  Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            # ============================================
            z = Variable(torch.randn(bs, z_dim)).cuda()
            r_imgs = Variable(imgs).cuda()
            f_imgs = G(z)

            # WGAN Loss
            loss_D = -torch.mean(D(r_imgs)) + torch.mean(D(f_imgs))
            # compute Gradient Penalty
            gradient_penalty = compute_gp(D, r_imgs, f_imgs)
            loss_D = loss_D + w_gp * gradient_penalty

            # Model backwarding
            D.zero_grad()
            loss_D.backward()

            # Update the discriminator.
            opt_D.step()

            """ Medium: Clip weights of discriminator. """
            # for p in D.parameters():
            #    p.data.clamp_(-clip_value, clip_value)

            # ============================================
            #  Train G
            #  Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            # ============================================

            # loss_G = -torch.mean(D(f_imgs))

            if steps % n_critic == 0:
                # Generate some fake images.
                z = Variable(torch.randn(bs, z_dim)).cuda()
                f_imgs = G(z)

                # Model forwarding
                # f_logit = D(f_imgs)

                """ Medium: Use WGAN Loss"""
                # Compute the loss for the generator.
                # loss_G = criterion(f_logit, r_label)
                # WGAN Loss
                loss_G = -torch.mean(D(f_imgs))
# gradient penlty
# tensorboard plot loss
                # Model backwarding
                G.zero_grad()
                loss_G.backward()

                # Update the generator.
                opt_G.step()

            steps += 1

            # Set the info of the progress bar
            #   Note that the value of the GAN loss is not directly related to
            #   the quality of the generated images.
            progress_bar.set_infos({
                'Loss_D': round(loss_D.item(), 4),
                'Loss_G': round(loss_G.item(), 4),
                'Epoch': e + 1,
                'Step': steps
            })

            tb.add_scalar("Loss/G", loss_G, steps)
            tb.add_scalar("Loss/D", loss_D, steps)

        G.eval()
        f_imgs_sample = (G(z_sample[:8]).data + 1.0) / 2.0

        # add figure to tensorboard
        loads, temperature = Data.decode_16users(f_imgs_sample, npoint, nuser)
        fig, axis = plt.subplots(8, 1)
        for i in range(loads.shape[2]):
            axis[i % 8].plot(loads[0, :, i])
        tb.add_figure('profiles', fig, e+1)

        # calculate load characteristics
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
                npeak = len(peaks[0])
                max_array.append(max.item())
                mean_array.append(mean.item())
                npeak_array.append(npeak)
                lf_array.append(mean.item() / max.item())
        # print(max_array)
        # print(mean_array)
        fig_lc = plt.figure(1)
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
        # plt.show()
        tb.add_figure('load characteristics', fig_lc, e+1)
        # ==================================================================

        # f_imgs_sample = G(z_sample).data
        f_imgs_T = f_imgs_sample[:, 3]
        f_imgs_T = f_imgs_T.reshape((f_imgs_T.shape[0], 1, npoint, nuser))
        f_imgs_sample = f_imgs_sample[:, 0:3]

        filename = os.path.join(log_dir, f'Epoch_{epoch + 1:03d}.jpg')
        torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
        print(f' | Save some samples to {filename}.')

        # Show generated images in
        grid_img = torchvision.utils.make_grid(f_imgs_sample[0:6].cpu(), nrow=6)
        tb.add_image("images/fake", grid_img, e + 1)
        tb.add_image("images/real", r_grid_img, e + 1)
        grid_T = torchvision.utils.make_grid(f_imgs_T[0:6].cpu(), nrow=6)
        tb.add_image("Other/T", grid_T, e + 1)


        if (e + 1) % 1 == 0 or e == 0:
            # Save the checkpoints.
            torch.save(G.state_dict(), os.path.join(ckpt_dir, 'G.pth'))
            torch.save(D.state_dict(), os.path.join(ckpt_dir, 'D.pth'))

            # test classifier
            acc = classifier.test_GAN_from_data(loads)
            tb.add_scalar("Classifier Acc", acc, e)

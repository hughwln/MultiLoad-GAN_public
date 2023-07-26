'''
Describe: Train SingleLoad-GAN
Author: Yi Hu
Email: yhu28@ncsu.edu
'''
import random
import torch
import numpy as np
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
# import singleUser_Data
import Data
from singleUser_GAN import *
from sys import platform

class CrypkoDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        # 1. Load the image
        img = torchvision.io.read_image(fname)
        # 2. Resize and normalize the images using torchvision.
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples

# ==================================
# compute gradient penalty
# ==================================
def compute_gp(netD, real_data, fake_data):
    batch_size = real_data.size(0)
    # Sample Epsilon from uniform distribution
    eps = torch.rand(batch_size, 1, 1).to(real_data.device)
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

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    same_seeds(2021)
    npoint = 96*7
    nuser = 1
    newRiverloader = Data.newRiverLoader()
    X = newRiverloader.load_data_for_GAN()
    X = X.iloc[:, 1:9].to_numpy()
    X = X.reshape((-1), order='F')
    # X = singleUser_Data.load_data()  # numpy  columns=1
    X = X.reshape((-1, npoint))
    X = torch.tensor(X).to(device, dtype=torch.float)
    # shuffle input data
    rand_idx = torch.randperm(X.shape[0])
    X = X[rand_idx]
    print(torch.max(X))
    # ======================================
    # Training
    # ======================================
    # Training hyperparameters
    batch_size = 64
    z_dim = 100
    z_sample = Variable(torch.randn(100, z_dim)).to(device, dtype=torch.float)
    lr = 1e-4
    # lr = 5e-5

    """ Medium: WGAN, 50 epoch, n_critic=5, clip_value=0.01 """
    n_epoch = 200  # 50
    n_critic = 1  # 5
    clip_value = 0.01
    w_gp = 10

    if platform == "linux" or platform == "linux2":
        workspace_dir = '~/Documents/Code/Research/MultiLoad-GAN_public/singleUser'
    elif platform == "darwin":
        workspace_dir = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\singleUser'
    elif platform == "win32":
        workspace_dir = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\singleUser'
    log_dir = os.path.join(workspace_dir, 'logs')
    ckpt_dir = os.path.join(workspace_dir, 'checkpoints')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Model
    G = Generator(in_dim=z_dim).to(device)
    D = Discriminator(1).to(device)
    G.train()
    D.train()

    # Loss
    criterion = nn.BCELoss()

    """ Medium: Use RMSprop for WGAN. """
    # Optimizer
    # opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    # opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.RMSprop(D.parameters(), lr=lr)
    opt_G = torch.optim.RMSprop(G.parameters(), lr=lr*1.2)

    dataloader = DataLoader(X, batch_size=batch_size, shuffle=True, num_workers=0)

    tb = SummaryWriter(f"tensorboard/log")

    # ===========================
    # Training loop
    # ===========================

    steps = 0
    for e, epoch in enumerate(range(n_epoch)):
        progress_bar = qqdm(dataloader)
        for i, data in enumerate(progress_bar):
            # print(data)
            imgs = data.reshape((data.shape[0], 1, npoint))
            imgs = imgs.to(device, dtype=torch.float)
            # transform from [0, 10] to [-1, 1]
            # imgs = (imgs-5.0) / 5.0
            imgs = (imgs-3.0) / 3.0
            bs = imgs.size(0)

            # ============================================
            #  Train D
            #  Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            # ============================================
            z = Variable(torch.randn(bs, z_dim)).to(device, dtype=torch.float)
            r_imgs = Variable(imgs).to(device)
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

            if steps % n_critic == 0:
                # Generate some fake images.
                z = Variable(torch.randn(bs, z_dim)).to(device, dtype=torch.float)
                f_imgs = G(z)

                # WGAN Loss
                loss_G = -torch.mean(D(f_imgs))

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
        f_imgs_sample = G(z_sample).data.cpu()
        # f_imgs_sample = f_imgs_sample * 5.0 + 5.0
        f_imgs_sample = f_imgs_sample * 3.0 + 3.0
        # f_imgs_sample = f_imgs_sample.reshape()

        # add figure to tensorboard
        fig, axis = plt.subplots(8, 1)
        for i in range(8):
            axis[i % 8].plot(f_imgs_sample[i, 0])
            # print(f_imgs_sample[i, 0])
        tb.add_figure('profiles', fig, e+1)

        G.train()

        if (e + 1) % 5 == 0 or e == 0:
            # Save the checkpoints.
            torch.save(G.state_dict(), os.path.join(ckpt_dir, 'G_single_1.pth'))
            torch.save(D.state_dict(), os.path.join(ckpt_dir, 'D_single_1.pth'))


'''
Describe: Build SingleLoad-GAN
Author: Yi Hu
Email: yhu28@ncsu.edu
'''
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

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
    Output shape: (N, 96*7) (N, 96*7)
    """

    def __init__(self, in_dim, dim=64):
        super(Generator, self).__init__()

        def dconv_bn_relu(in_dim, out_dim, k, s, p, output_p):
            return nn.Sequential(
                nn.ConvTranspose1d(in_dim, out_dim, k, s,
                                   padding=p, output_padding=output_p, bias=False),
                nn.BatchNorm1d(out_dim),
                nn.ReLU()
            )

        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4),    # 4*1
            nn.ReLU()
        )
        # 5 2 2 1           *2
        # 5 1 2 0           *1
        # 3 3 0 0           *3
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4, 5, 2, 2, 1),    # 8
            dconv_bn_relu(dim * 4, dim * 2, 5, 2, 2, 1),    # 16
            dconv_bn_relu(dim * 2, dim, 5, 2, 2, 1),        # 32
            dconv_bn_relu(dim, dim, 3, 3, 0, 0),  # 96
            nn.ConvTranspose1d(dim, 1, 7, 7, padding=0, output_padding=0),  # 96*7
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4)
        y = self.l2_5(y)
        return y


class Discriminator(nn.Module):
    """
    Input shape: (N, 96*7)
    Output shape: (N, )
    """

    def __init__(self, in_dim, dim=32):
        super(Discriminator, self).__init__()

        def conv_bn_lrelu(in_dim, out_dim, k, s, p=0):
            return nn.Sequential(
                # nn.Conv1d(in_dim, out_dim, 5, 2, 2),
                nn.Conv1d(in_dim, out_dim, k, s, p),
                nn.BatchNorm1d(out_dim),
                nn.LeakyReLU(0.2),
            )

        """ Medium: Remove the last sigmoid layer for WGAN. """
        self.ls = nn.Sequential(
            nn.Conv1d(in_dim, dim, 7, 7, 0),     # 96
            nn.LeakyReLU(0.2),
            conv_bn_lrelu(dim, dim, 3, 3, 0),  # 32
            conv_bn_lrelu(dim, dim * 2, 5, 2, 2),        # 16
            conv_bn_lrelu(dim * 2, dim * 4, 5, 2, 2),    # 8
            conv_bn_lrelu(dim * 4, dim * 8, 5, 2, 2),    # 4
            nn.Conv1d(dim * 8, 1, 4),           # 1
            # nn.Sigmoid(),
        )
        self.apply(weights_init)

    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y

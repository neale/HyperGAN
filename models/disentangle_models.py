import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
import utils


class DiscBase(nn.Module):
    def __init__(self, args):
        super(DiscBase, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        
        self.name = 'DiscBase'
        self.linear1 = nn.Linear(self.z, 256)
        self.linear2 = nn.Linear(256, 512)
        self.linear3 = nn.Linear(512, 512)
        self.relu = nn.ELU(inplace=True)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(512)

    def forward(self, x):
        # print ('Dz in: ', x.shape)
        x = x.view(self.batch_size, -1)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.relu(self.bn3(self.linear3(x)))
        # print ('Dz out: ', x.shape)
        return x


class DiscHead(nn.Module):
    def __init__(self, args):
        super(DiscHead, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        
        self.name = 'DiscHead'
        self.linear1 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        return x


class DiscQ(nn.Module):
    def __init__(self, args):
        super(DiscQ, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        
        self.name = 'DiscriminatorZ'
        self.linear1 = nn.Linear(512, 256)
        self.linear_mu = nn.Linear(256, 2)
        self.linear_var = nn.Linear(256, 2)
        self.linear_disc = nn.Linear(256, 10)
        self.sigmoid = nn.Sigmoid()
        self.bn2 = nn.BatchNorm1d(256)


    def forward(self, x):
        # print ('Dz in: ', x.shape)
        x = self.linear1(x)
        mu = self.linear_mu(x).squeeze()
        var = self.linear_var(x).squeeze()
        x = self.linear_disc(x).squeeze()
        # print ('Dz out: ', x.shape)
        return mu, var, x

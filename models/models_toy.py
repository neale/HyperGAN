import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
import utils


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'Encoder'
        self.linear1 = nn.Linear(self.ze, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, self.z*2)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu = nn.ELU(inplace=True)

    
    def forward(self, x):
        #print ('E in: ', x.shape)
        x = x.view(-1, self.ze) #flatten filter size
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        x = x.view(-1, 3, self.z)
        w1 = x[:, 0]
        w2 = x[:, 1]
        #print ('E out: ', x.shape)
        return w1, w2, w3

""" Linear (20 x 200) """
class GeneratorW1(nn.Module):
    def __init__(self, args):
        super(GeneratorW3, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW3'
        self.linear1 = nn.Linear(self.z, 256)
        self.linear2 = nn.Linear(256, 4000)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        #print ('W3 in : ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 20, 200)
        #print ('W3 out: ', x.shape)
        return x


""" Linear (20 x 1) """
class GeneratorW1(nn.Module):
    def __init__(self, args):
        super(GeneratorW3, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW3'
        self.linear1 = nn.Linear(self.z, 128)
        self.linear2 = nn.Linear(128, 20)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        #print ('W3 in : ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 1, 20)
        #print ('W3 out: ', x.shape)
        return x


class DiscriminatorZ(nn.Module):
    def __init__(self, args):
        super(DiscriminatorZ, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        
        self.name = 'DiscriminatorZ'
        self.linear1 = nn.Linear(self.z, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 1)
        self.relu = nn.ELU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print ('Dz in: ', x.shape)
        x = x.view(self.batch_size, -1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        x = self.sigmoid(x)
        # print ('Dz out: ', x.shape)
        return x

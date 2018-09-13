import numpy as np
import torch

from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'Encoder'
        self.linear1 = nn.Linear(self.ze, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, self.z*3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
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
        w3 = x[:, 2]
        #print ('E out: ', x.shape)
        return w1, w2, w3


class GeneratorW1(nn.Module):
    def __init__(self, args):
        super(GeneratorW1, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW1'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 800)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        # print ('W1 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 32, 1, 5, 5)
        #print ('W1 out: ', x.shape)
        return x


""" Convolutional (32 x 32 x 5 x 5) """
class GeneratorW2(nn.Module):
    def __init__(self, args):
        super(GeneratorW2, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW2'
        input_dim = args.z + 10# + args.factors
        self.linear1 = nn.Linear(input_dim, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.linear3 = nn.Linear(2048, 4096)
        self.linear4 = nn.Linear(4096, 25600)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(2048)
        self.bn3 = nn.BatchNorm1d(4096)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x, y):#, c):
        # print ('W2 x: ', x.shape, 'c:', c.shape)
        x = torch.cat((x, y), -1)#, c), -1)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.relu(self.bn3(self.linear3(x)))
        x = self.linear4(x)
        x = x.view(-1, 32, 32, 5, 5)
        #print ('W2 out: ', x.shape)
        return x


""" Linear (512 x 10) """
class GeneratorW3(nn.Module):
    def __init__(self, args):
        super(GeneratorW3, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW3'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 512*10)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        #print ('W3 in : ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 10, 512)
        #print ('W3 out: ', x.shape)
        return x


class DiscriminatorQ(nn.Module):
    def __init__(self, args):
        super(DiscriminatorQ, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        
        self.name = 'DiscriminatorQ'
        self.linear1 = nn.Linear(800, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.linear3 = nn.Linear(2048, 5000)
        self.relu = nn.LeakyReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(1024, .8)
        self.bn2 = nn.BatchNorm1d(2048, .8)
        self.bn3 = nn.BatchNorm1d(5000, .8)
        self.dropout = nn.Dropout2d(0.25)
        self.softmax = nn.Softmax(1)
        
        self.adv_layer = nn.Linear(5000, 1)
        self.aux_layer = nn.Linear(5000, 10)
        self.factor_layer = nn.Linear(5000, args.factors)
    
    def forward(self, x):
        # print ('Dz in: ', x.shape)
        x = x.view(self.batch_size, -1)
        x = self.relu(self.linear1(x))
        #x = self.bn1(self.dropout(x))
        x = self.relu(self.linear2(x))
        #x = self.bn2(self.dropout(x))
        x = self.relu(self.linear3(x))
        #x = self.bn3(self.dropout(x))
        # print ('Dz out: ', x.shape)
        
        disc = self.adv_layer(x)
        label = self.softmax(self.aux_layer(x))
        factors = self.factor_layer(x)
        return disc, label, factors


class DiscriminatorZ(nn.Module):
    def __init__(self, args):
        super(DiscriminatorZ, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        
        self.name = 'DiscriminatorZ'
        self.linear1 = nn.Linear(self.z, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 1)
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


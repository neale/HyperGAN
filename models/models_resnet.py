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


class Encoderz(nn.Module):
    def __init__(self, args):
        super(Encoderz, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'Encoderz'
        self.linear1 = nn.Linear(self.ze, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, self.z*3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        #print ('E in: ', x.shape)
        x = x.view(-1, self.ze) #flatten filter size
        x = torch.zeros_like(x).normal_(0, 0.01) + x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        x = x.view(-1, 3, self.z)
        w1 = x[:, 0]
        w2 = x[:, 1]
        w3 = x[:, 2]
        #print ('E out: ', x.shape)
        return w1, w2, w3


""" (3, 64, 7, 7) """
class GeneratorW1(nn.Module):
    def __init__(self, args):
        super(GeneratorW1, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW1'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 3*64*7*7)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        # print ('W1 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 3, 64, 7, 7)
        #print ('W1 out: ', x.shape)
        return x

""" (64, 64, 3, 3) """
class GeneratorW2(nn.Module):
    def __init__(self, args):
        super(GeneratorW2, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW2'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 64*64*3*3)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        # print ('W1 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 64, 64, 3, 3)
        #print ('W1 out: ', x.shape)
        return x

""" (64, 64, 3, 3) """
class GeneratorW3(nn.Module):
    def __init__(self, args):
        super(GeneratorW3, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW3'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 64*64*3*3)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        # print ('W1 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 64, 64, 3, 3)
        #print ('W1 out: ', x.shape)
        return x


""" (64, 64, 3, 3) """
class GeneratorW4(nn.Module):
    def __init__(self, args):
        super(GeneratorW4, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW4'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 64*64*3*3)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        # print ('W1 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 64, 64, 3, 3)
        #print ('W1 out: ', x.shape)
        return x

""" (64, 64, 3, 3) """
class GeneratorW5(nn.Module):
    def __init__(self, args):
        super(GeneratorW5, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW5'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 64*64*3*3)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        # print ('W1 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 64, 64, 3, 3)
        #print ('W1 out: ', x.shape)
        return x

""" (64, 128, 3, 3) """
class GeneratorW6(nn.Module):
    def __init__(self, args):
        super(GeneratorW6, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW6'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 64*128*3*3)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        # print ('W1 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 64, 128, 3, 3)
        #print ('W1 out: ', x.shape)
        return x

""" (128, 128, 3, 3) """
class GeneratorW7(nn.Module):
    def __init__(self, args):
        super(GeneratorW7, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW7'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 128*128*3*3)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        # print ('W1 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 128, 128, 3, 3)
        #print ('W1 out: ', x.shape)
        return x

""" (64, 128, 3, 3) """
class GeneratorW8(nn.Module):
    def __init__(self, args):
        super(GeneratorW8, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW8'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 64*128*3*3)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        # print ('W1 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 64, 128, 3, 3)
        #print ('W1 out: ', x.shape)
        return x

""" (128, 128, 3, 3) """
class GeneratorW9(nn.Module):
    def __init__(self, args):
        super(GeneratorW9, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW9'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 128*128*3*3)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        # print ('W1 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 128, 128, 3, 3)
        #print ('W1 out: ', x.shape)
        return x

""" (128, 128, 3, 3) """
class GeneratorW10(nn.Module):
    def __init__(self, args):
        super(GeneratorW10, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW10'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 128*128*3*3)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        # print ('W1 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 128, 128, 3, 3)
        #print ('W1 out: ', x.shape)
        return x

""" (128, 256, 3, 3) """
class GeneratorW11(nn.Module):
    def __init__(self, args):
        super(GeneratorW11, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW11'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 128*256*3*3)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        # print ('W1 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 128, 256, 3, 3)
        #print ('W1 out: ', x.shape)
        return x

""" Convolutional (256 x 256 x 3 x 3) """
class GeneratorW12(nn.Module):
    def __init__(self, args):
        super(GeneratorW12, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW12'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 256*256*3*3)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        #print ('W2 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 256, 256, 3, 3)
        #print ('W2 out: ', x.shape)
        return x

""" Convolutional (128 x 256 x 3 x 3) """
class GeneratorW13(nn.Module):
    def __init__(self, args):
        super(GeneratorW13, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW13'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 128*256*3*3)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        #print ('W2 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 128, 256, 3, 3)
        #print ('W2 out: ', x.shape)
        return x

""" Convolutional (256 x 256 x 3 x 3) """
class GeneratorW14(nn.Module):
    def __init__(self, args):
        super(GeneratorW14, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW14'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 256*256*3*3)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        #print ('W2 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 256, 256, 3, 3)
        #print ('W2 out: ', x.shape)
        return x

""" Convolutional (256 x 256 x 3 x 3) """
class GeneratorW15(nn.Module):
    def __init__(self, args):
        super(GeneratorW15, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW15'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 256*256*3*3)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        #print ('W2 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 256, 256, 3, 3)
        #print ('W2 out: ', x.shape)
        return x

""" Convolutional (256 x 512 x 3 x 3) """
class GeneratorW16(nn.Module):
    def __init__(self, args):
        super(GeneratorW16, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW16'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 256*512*3*3)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        #print ('W2 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 256, 512, 3, 3)
        #print ('W2 out: ', x.shape)
        return x

""" Convolutional (512 x 512 x 3 x 3) """
class GeneratorW17(nn.Module):
    def __init__(self, args):
        super(GeneratorW17, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW17'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 512*512*3*3)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        #print ('W2 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 512, 512, 3, 3)
        #print ('W2 out: ', x.shape)
        return x

""" Convolutional (256 x 512 x 3 x 3) """
class GeneratorW18(nn.Module):
    def __init__(self, args):
        super(GeneratorW18, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW18'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 256*512*3*3)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        #print ('W2 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 256, 512, 3, 3)
        #print ('W2 out: ', x.shape)
        return x

""" Convolutional (512 x 512 x 3 x 3) """
class GeneratorW19(nn.Module):
    def __init__(self, args):
        super(GeneratorW19, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW19'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 512*512*3*3)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        #print ('W2 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 512, 512, 3, 3)
        #print ('W2 out: ', x.shape)
        return x

""" Convolutional (512 x 512 x 3 x 3) """
class GeneratorW20(nn.Module):
    def __init__(self, args):
        super(GeneratorW20, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW20'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 512*512*3*3)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        #print ('W2 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 512, 512, 3, 3)
        #print ('W2 out: ', x.shape)
        return x

""" Linear (512 x 1000) """
class GeneratorW21(nn.Module):
    def __init__(self, args):
        super(GeneratorW21, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW21'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 512*1000)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        #print ('W3 in : ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 1000, 512)
        #print ('W3 out: ', x.shape)
        return x


class DiscriminatorZ(nn.Module):
    def __init__(self, args):
        super(DiscriminatorZ, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        
        self.name = 'DiscriminatorZ'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 1)
        self.relu = nn.ReLU(inplace=True)
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

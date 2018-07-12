import numpy as np
from torch import nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, args, datashape):
        super(Discriminator, self).__init__()
        self.shape = datashape[::-1]
        self.dim = args.dim
        self.conv1 = nn.Conv2d(self.shape[-1], self.dim, 3, stride=2, padding=2)
        self.conv2 = nn.Conv2d(self.dim, 2*self.dim, 3, stride=2, padding=2)
        self.conv3 = nn.Conv2d(2*self.dim, 4*self.dim, 3, stride=2, padding=2)
        self.output = nn.Linear(4*4*self.dim, 1)

    def forward(self, input):
        input = input.view(-1, *self.shape[::-1]) # weirdest transpose ever
        out = self.conv1(input)
        out = F.relu(F.dropout(out, 0.3), inplace=True)
        out = self.conv2(out)
        out = F.relu(F.dropout(out, 0.3), inplace=True)
        out = self.conv3(out)
        out = F.relu(F.dropout(out, 0.3), inplace=True)
        out = out.view(-1, 4*4*self.dim)
        out = self.output(out)
        return out


class DiscriminatorWide(nn.Module):
    def __init__(self, args, datashape):
        super(DiscriminatorWide, self).__init__()
        self.shape = datashape
        self.dim = args.dim
        self.conv1 = nn.Conv2d(self.shape[-1], 4*self.dim, 3, stride=2, padding=2)
        self.conv2 = nn.Conv2d(4*self.dim, 8*self.dim, 3, stride=2, padding=2)
        self.conv3 = nn.Conv2d(8*self.dim, 16*self.dim, 3, stride=2, padding=2)
        self.output = nn.Linear(4*4*4*self.dim, 1)

    def forward(self, input):
        # print ('d in', input.size())
        input = input.view(-1, *self.shape[::-1]) # weirdest transpose ever
        out = self.conv1(input)
        out = F.relu(F.dropout(out, 0.3), inplace=True)
        out = self.conv2(out)
        out = F.relu(F.dropout(out, 0.3), inplace=True)
        out = self.conv3(out)
        out = F.relu(F.dropout(out, 0.3), inplace=True)
        out = out.view(-1, 4*4*4*self.dim)
        out = self.output(out)
        # print ('d out: ', out.size())
        return out


class ResNetDiscriminator(nn.Module):
    def __init__(self, args, datashape):
        super(ResNetDiscriminator, self).__init__()
        self.shape = datashape
        self.dim = args.dim
        conv1 = nn.Conv2d(self.shape[-1], 4*self.dim, 3, stride=2, padding=2)
        conv2 = nn.Conv2d(4*self.dim, 8*self.dim, 3, stride=2, padding=2)
        conv3 = nn.Conv2d(8*self.dim, 16*self.dim, 3, stride=2, padding=2)
        output = nn.Linear(4*4*4*self.dim, 1)

        self.conv1 = conv1
        self.conv2 = conv2 
        self.conv3 = conv3
        self.output = output

    def forward(self, x):
        # print ('d in', x.size())
        x = x.view(-1, *self.shape[::-1]) # weirdest transpose ever
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = x.view(-1, 4*4*4*self.dim)
        x = self.output(x)
        # print ('d out: ', x.size())
        return x


class DiscriminatorWide7(nn.Module):
    def __init__(self, args, datashape):
        super(DiscriminatorWide7, self).__init__()
        self.shape = datashape
        self.dim = args.dim
        self.conv1 = nn.Conv2d(self.shape[-1], 4*self.dim, 3, stride=2, padding=2)
        self.conv2 = nn.Conv2d(4*self.dim, 8*self.dim, 3, stride=2, padding=2)
        self.conv3 = nn.Conv2d(8*self.dim, 16*self.dim, 3, stride=2, padding=2)
        self.output = nn.Linear(4*4*4*self.dim, 1)

    def forward(self, input):
        # print ('d in', input.size())
        input = input.view(-1, *self.shape[::-1]) # weirdest transpose ever
        out = self.conv1(input)
        out = F.relu(F.dropout(out, 0.3), inplace=True)
        out = self.conv2(out)
        out = F.relu(F.dropout(out, 0.3), inplace=True)
        out = self.conv3(out)
        out = F.relu(F.dropout(out, 0.3), inplace=True)
        out = out.view(-1, 4*4*4*self.dim)
        out = self.output(out)
        # print ('d out: ', out.size())
        return out


class DiscriminatorFC(nn.Module):
    def __init__(self, args):
        super(DiscriminatorFC, self).__init__()
        self.shape = (1, 3, 3)
        self.dim = 64
        block = nn.Sequential(
                nn.Linear(9, self.dim),
                nn.Dropout(p=0.3),
                nn.ReLU(True),
                nn.Linear(self.dim, 4*self.dim),
                nn.Dropout(p=0.3),
                nn.ReLU(True),
                nn.Linear(4*self.dim, 4*self.dim),
                nn.Dropout(p=0.3),
                nn.ReLU(True),
                nn.Linear(4*self.dim, 8*self.dim),
                nn.Dropout(p=0.3),
                nn.ReLU(True),
                nn.Linear(8*self.dim, 2*self.dim),
                nn.ReLU(True),
                )
        self.main = block
        self.output = nn.Linear(2*self.dim, 1)

    def forward(self, input):
        input = input.view(-1, 1, 9)
        out = self.main(input)
        out = out.view(-1, 2*self.dim)
        out = self.output(out)
        return out.view(-1)


class DiscriminatorWide7FC(nn.Module):
    def __init__(self, args, datashape):
        super(DiscriminatorWide7FC, self).__init__()
        self.dim = dim = args.dim
        self.nf = nf = 512
        self.dshape = dshape = datashape
        
        self.ng = ng = dshape[-1]*dshape[-2]*dshape[0]

        self.linear1 = nn.Linear(ng, nf)
        self.linear2 = nn.Linear(nf, nf)
        self.linear3 = nn.Linear(nf, 1)
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print ('D in: ', x.shape)
        x = x.view(-1, self.ng)
        x = self.elu(self.linear1(x))
        x = self.elu(self.linear2(x))
        x = self.elu(self.linear3(x))
        x = self.sigmoid(x)
        # print ('D out: ', x.shape)
        return x

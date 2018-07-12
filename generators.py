import numpy as np
from torch import nn
from torch.nn import functional as F


class MGenerator(nn.Module):
    def __init__(self, args):
        super(MGenerator, self).__init__()
        self.dim = 64
        self.shape = (64, 3, 3)
        preprocess = nn.Sequential(
                nn.Linear(self.dim, 2*self.dim),
                nn.BatchNorm2d(2*self.dim),
                nn.LeakyReLU(True),
                )
        block1 = nn.Sequential(
                nn.ConvTranspose2d(2*self.dim, self.dim, 3, stride=1),
                nn.BatchNorm2d(self.dim),
                nn.LeakyReLU(True),
                )
        self.block1 = block1
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 2*self.dim, 1, 1)
        output = self.block1(output)
        output = self.tanh(output)*4
        output.view(self.shape[0], -1, self.shape[1], self.shape[2])
        return output


class MGeneratorWide(nn.Module):
    def __init__(self, args, datashape):
        super(MGeneratorWide, self).__init__()
        self.dim = 64
        self.shape = datashape
        self.ndim = ndim = datashape[0] // self.dim
        self.preprocess = nn.Linear(self.dim, 7*7*self.dim)
        self.conv1 = nn.Conv2d(1, self.dim, 3, stride=1)
        self.conv2 = nn.Conv2d(self.dim, 2*self.dim, 3, stride=2)
        self.conv3 = nn.Conv2d(2*self.dim, 2*self.dim, 3, stride=2)
        self.conv4 = nn.Conv2d(2*self.dim, 4*self.dim, 3, stride=2)
        self.conv5 = nn.Conv2d(4*self.dim, 4*self.dim, 3, stride=1)
        self.conv6 = nn.Conv2d(4*self.dim, 4*self.dim, 1, stride=1)
        self.tanh = nn.Tanh()

    def forward(self, z):
        #print ('g in: ', z.size())
        x = self.preprocess(z)
        x = x.view(-1, 1, 56, 56)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        x = self.tanh(x)
        # print ('g out: ', x.size())
        return x


class MGeneratorWide7(nn.Module):
    def __init__(self, args, datashape):
        super(MGeneratorWide7, self).__init__()
        self.nf = nf = args.nf
        self.shape = datashape
        self.ndim = ndim = args.dim
        self.block1 = nn.Sequential(
                nn.ConvTranspose2d(ndim, nf*8, 3, 1, 1),
                nn.BatchNorm2d(nf*8),
                nn.ELU(True),
                )
        self.block2 = nn.Sequential(
                nn.ConvTranspose2d(nf*8, nf*4, 3, 1, 1),
                nn.BatchNorm2d(nf*4),
                nn.ELU(True),
                )
        self.block3 = nn.Sequential(
                nn.ConvTranspose2d(nf*4, nf*2, 3, 1, 1),
                nn.BatchNorm2d(nf*8),
                nn.ELU(True),
                )
        self.deconv_out = nn.ConvTranspose2d(nf*2, nf, 1)

    def forward(self, x):
        print ('G in: ', x.shape)
        x = self.block1(x)
        # print ('g block1 out : ', x.shape)
        x = self.block2(x)
        #print ('g block2 out : ', x.shape)
        x = self.block3(x)
        #print ('g block3 out : ', x.shape)
        x = self.deconv_out(x)
        print ('G out: ', x.shape)
        return output


class CGenerator(nn.Module):
    def __init__(self, args):
        super(CGenerator, self).__init__()
        self.dim = 64
        self.shape = (128, 3, 3)
        preprocess = nn.Sequential(
                nn.Linear(self.dim, 2*2*self.dim),
                nn.BatchNorm2d(2*2*self.dim),
                nn.LeakyReLU(True),
                )
        block1 = nn.Sequential(
                nn.ConvTranspose2d(2*2*self.dim, 2*self.dim, 3, stride=1),
                nn.BatchNorm2d(2*self.dim),
                nn.LeakyReLU(True),
                #nn.ReLU(True),
                )
        self.block1 = block1
        self.preprocess = preprocess
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 2*2*self.dim, 1, 1)
        output = self.block1(output)
        output = self.tanh(output)*4
        output.view(self.shape[0], -1, self.shape[1], self.shape[2])
        return output


class ResNetGenerator(nn.Module):
    def __init__(self, args):
        super(ResNetGenerator, self).__init__()
        self.dim = 64
        self.shape = (512, 3, 3)
        self.preprocess = nn.Linear(self.dim, 4*4*4*3*self.dim)
        self.conv1 = nn.Conv2d(3, self.dim, 3)
        self.conv2 = nn.Conv2d(self.dim, self.dim, 3, stride=2)
        self.conv3 = nn.Conv2d(self.dim, 2*self.dim, 3)
        self.conv4 = nn.Conv2d(2*self.dim, 2*self.dim, 3, stride=2)
        self.conv5 = nn.Conv2d(2*self.dim, 4*self.dim, 3)
        self.conv6 = nn.Conv2d(4*self.dim, 4*self.dim, 3, stride=2) #256
        self.conv7 = nn.Conv2d(4*self.dim, 8*self.dim, 3)
        self.conv8 = nn.Conv2d(8*self.dim, 8*self.dim, 1) #512
        
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.preprocess(x)
        x = x.view(-1, 3, 64, 64)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        x = F.leaky_relu(self.conv7(x))
        x = F.leaky_relu(self.conv8(x))
        out = self.tanh(x)
        return out


class GeneratorWide7FC(nn.Module):
    def __init__(self, args, datashape):
        super(GeneratorWide7FC, self).__init__()
        self.dim = dim = args.dim
        self.dshape = datashape
        self.nu = nlinear = 512
        self.nf = nconv = datashape[-1]
        self.nc = nc = datashape[0]

        self.linear1 = nn.Linear(nlinear, nlinear*2)
        self.linear2 = nn.Linear(nlinear*2, nlinear*4)
        self.linear3 = nn.Linear(nlinear*4, nc)
        self.linear_out = nn.Linear(nc, nc * nfilters * nfilters)
        self.elu = nn.ELU()
        self.bn1 = nn.BatchNorm2d(nlinear*2)
        self.bn2 = nn.BatchNorm2d(nlinear*4)
        self.bn3 = nn.BatchNorm2d(nc)

    def forward(self, x):
        print ('G in: ', x.shape)
        x = self.elu(self.bn1(self.linear1(x)))
        x = self.elu(self.bn2(self.linear2(x)))
        x = self.elu(self.bn3(self.linear3(x)))
        x = self.linear_out(x)
        x = x.view(datashape)
        print ('G out: ', x.shape)
        return x

class GLayer7FC(nn.Module):
    def __init__(self, args, datashape):
        super(GLayer7FC, self).__init__()
        self.dim = dim = args.dim
        self.dshape = datashape
        self.nf = nf = datashape[-1]
        self.nc = nc = datashape[0]

        self.linear1 = nn.Linear(dim, nf*nf*8)
        self.linear2 = nn.Linear(nf*nf*8, nf*nf*4)
        self.linear3 = nn.Linear(nf*nf*4, nf*nf*2)
        self.linear_out = nn.Linear(nf*nf*2, nf*nf)
        self.elu = nn.ELU()
        self.bn1 = nn.BatchNorm2d(nf*nf*8)
        self.bn2 = nn.BatchNorm2d(nf*nf*4)
        self.bn3 = nn.BatchNorm2d(nf*nf*2)

    def forward(self, x):
        # print ('G in: ', x.shape)
        x = self.elu(self.bn1(self.linear1(x)))
        x = self.elu(self.bn2(self.linear2(x)))
        x = self.elu(self.bn3(self.linear3(x)))
        x = self.linear_out(x)
        x = x.view(-1, self.nf, self.nf)
        # print ('G out: ', x.shape)
        return x



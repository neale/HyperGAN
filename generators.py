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
        preprocess = nn.Sequential(
                nn.Linear(self.dim, ndim*2*self.dim),
                nn.BatchNorm2d(ndim*2*self.dim),
                nn.LeakyReLU(True),
                )
        block1 = nn.Sequential(
                nn.ConvTranspose2d(ndim*2*self.dim, ndim*self.dim, 3, stride=1),
                nn.BatchNorm2d(ndim*self.dim),
                nn.LeakyReLU(True),
                )
        self.block1 = block1
        self.preprocess = preprocess
        self.tanh = nn.Tanh()

    def forward(self, z):
        # print ('g in: ', z.size())
        output = self.preprocess(z)
        output = output.view(-1, self.ndim*2*self.dim, 4, 4)
        output = self.block1(output)
        output = self.tanh(output)*4
        output.view(self.shape[0], -1, self.shape[1], self.shape[2])
        # print ('g out: ', output.size())
        return output


class MGeneratorWide7(nn.Module):
    def __init__(self, args, datashape):
        super(MGeneratorWide7, self).__init__()
        self.dim = 64
        self.shape = datashape
        self.ndim = ndim = datashape[0] // self.dim
        preprocess = nn.Sequential(
                nn.Linear(self.dim, ndim*3*3*3*self.dim),
                #nn.BatchNorm2d(ndim*3*3*3*self.dim),
                nn.LeakyReLU(True),
                )
        block1 = nn.Sequential(
                nn.ConvTranspose2d(ndim*3*self.dim, ndim*self.dim, 5, stride=1, padding=1),
                #nn.BatchNorm2d(ndim*self.dim),
                nn.LeakyReLU(True),
                )
        block2 = nn.Sequential(
                nn.ConvTranspose2d(ndim*self.dim, ndim*self.dim, 5, stride=1, padding=1),
                # nn.BatchNorm2d((ndim*self.dim)),
                nn.LeakyReLU(True),
                )
        deconv_out = nn.ConvTranspose2d(ndim*self.dim, ndim*self.dim, 1, stride=1)
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.tanh = nn.Tanh()

    def forward(self, z):
        # print ('g in: ', z.size())
        output = self.preprocess(z)
        output = output.view(-1, self.ndim*3*self.dim, 3, 3)
        # print ('g pp out: ', output.size())
        output = self.block1(output)
        # print ('g block1 out : ', output.size())
        output = self.block2(output)
        #print ('g block2 out : ', output.size())
        #output = self.deconv_out(output)
        #print ('deconv out : ', output.size())
        output = self.tanh(output)
        output.view(self.shape[0], -1, self.shape[1], self.shape[2])
        #print ('g out: ', output.size())
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


class GeneratorFC(nn.Module):
    def __init__(self, args):
        super(GeneratorFC, self).__init__()
        self.dim = 64
        self.in_shape = 64
        preprocess = nn.Sequential(
                nn.Linear(self.dim, 4*self.dim),
                nn.BatchNorm1d(4*self.dim),
                nn.ReLU(True),
                )
        block = nn.Sequential(
                nn.Linear(4*self.dim, 4*self.dim,),
                nn.BatchNorm1d(4*self.dim),
                nn.ReLU(True),
                )
        block2 = nn.Sequential(
                nn.Linear(4*self.dim, 8*self.dim,),
                nn.BatchNorm1d(8*self.dim),
                nn.ReLU(True),
                )
        out = nn.Sequential(
                nn.Linear(8*self.dim, 2*self.dim),
                nn.ReLU(True),
                nn.Linear(2*self.dim, 9)
                )
        self.block = block
        self.block2 = block2
        self.out = out
        self.preprocess = preprocess
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4*self.dim)
        output = self.block(output)
        # output = self.block(output)
        output = self.block2(output)
        output = self.out(output)
        output = self.tanh(output)*4
        return output.view(-1, 9)

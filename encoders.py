import numpy as np
from torch import nn


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.shape = (1, 3, 3)
        self.dim = 3
        convblock = nn.Sequential(
                nn.Conv2d(1, self.dim, 1, stride=2, padding=1),
                nn.Dropout(p=0.3),
                nn.ReLU(True),
                nn.Conv2d(self.dim, 2*self.dim, 1, stride=1, padding=2),
                nn.Dropout(p=0.3),
                nn.ReLU(True),
                nn.Conv2d(2*self.dim, 4*self.dim, 1, stride=1, padding=2),
                nn.Dropout(p=0.3),
                nn.ReLU(True),
                )
        self.main = convblock
        self.output = nn.Linear(4*4*4*self.dim, self.dim)

    def forward(self, input):
        input = input.view(-1, 1, 3, 3)
        out = self.main(input)
        out = out.view(-1, 4*4*4*self.dim)
        out = self.output(out)
        return out.view(-1, self.dim)

class EncoderWide7(nn.Module):
    def __init__(self, args, datashape, is_training=True):
        super(MGeneratorWide7, self).__init__()
        self.dim = dim = args.dim
        self.shape = datashape
        self.nf = nf = 128
        self.nc = nc = datashape[1]

        self.block1 = nn.Sequential(
                nn.Conv2d(nc, nf, 3, 2, 1),
                nn.BatchNorm2d(nf),
                nn.LeakyReLU(True),
                )
        self.block2 = nn.Sequential(
                nn.Conv2d(nf, nf*2, 3, 2, 1),
                nn.BatchNorm2d((nf*2)),
                nn.LeakyReLU(True),
                )
        self.conv_out = nn.Conv2d(nf*2, dim, 3, 2)

    def forward(self, x):
        print ('E in: ', x.shape)
        if self.is_training;
            z = torch.normal(torch.zeros_like(x.data), std=0.01)
            x.data += z
        x = self.block1(x)
        # print ('g block1 out : ', x.shape)
        x = self.block2(x)
        #print ('g block2 out : ', x.shape)
        x = self.conv_out(x)
        print ('E out: ', x.shape)
        return x



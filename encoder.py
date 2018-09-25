import sys
import torch
import torch.nn as nn
import pprint
import argparse
import numpy as np

from torch import optim
from torch.nn import functional as F

import ops
import utils
import netdef
import datagen


def load_args():

    parser = argparse.ArgumentParser(description='param-wgan')
    parser.add_argument('--z', default=8, type=int, help='latent space width')
    parser.add_argument('--ze', default=100, type=int, help='encoder dimension')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=200000, type=int)
    parser.add_argument('--target', default='small2', type=str)
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--beta', default=1000, type=int)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--use_x', default=False, type=bool)
    parser.add_argument('--pretrain_e', default=False, type=bool)
    parser.add_argument('--scratch', default=False, type=bool)
    parser.add_argument('--exp', default='0', type=str)
    parser.add_argument('--use_d', default=False, type=str)
    parser.add_argument('--model', default='small', type=str)

    args = parser.parse_args()
    return args

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'Encoder'
        self.linear1 = nn.Linear(self.ze, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 1024)
        self.linear4 = nn.Linear(1024, self.z*3)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        #print ('E in: ', x.shape)
        x = x.view(-1, self.ze) #flatten filter size
        x = torch.zeros_like(x).normal_(0, 0.01) + x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.relu(self.bn3(self.linear3(x)))
        x = self.linear4(x)
        return x
        x = x.view(-1, 3, self.z)
        w1 = x[:, 0]
        w2 = x[:, 1]
        w3 = x[:, 2]
        #print ('E out: ', x.shape)
        return w1, w2, w3

# example for some module
def train(args):
    
    torch.manual_seed(8734)
    netE = Encoder(args).cuda()

    optimE = optim.Adam(netE.parameters(), lr=1e-4, betas=(0.5, 0.9))
    
    x_dist = utils.create_d(args.ze)
    z_dist = utils.create_d(args.z*3)
    one = torch.FloatTensor([1]).cuda()
    mone = (one * -1).cuda()
    print ("==> pretraining encoder")
    j = 0
    final = 100.
    e_batch_size = 1000
    for j in range(2000):
        x = utils.sample_d(x_dist, e_batch_size)
        z = utils.sample_d(z_dist, e_batch_size)
        code = netE(x)
        #code = code.view(e_batch_size, args.z)
        mean_loss, cov_loss = ops.pretrain_loss(code, z)
        loss = mean_loss + cov_loss
        loss.backward()
        optimE.step()
        netE.zero_grad()
        print ('Pretrain Enc iter: {}, Mean Loss: {}, Cov Loss: {}'.format(
            j, mean_loss.item(), cov_loss.item()))
        final = loss.item()
        if loss.item() < 0.1:
            print ('Finished Pretraining Encoder')
            break

if __name__ == '__main__':

    args = load_args()
    if args.model == 'small':
        import models.models_mnist_small as models
    elif args.model == 'nobn':
        import models.models_mnist_nobn as models
    elif args.model == 'full':
        import models.models_mnist as models
    else:
        raise NotImplementedError

    modeldef = netdef.nets()[args.target]
    pprint.pprint (modeldef)
    # log some of the netstat quantities so we don't subscript everywhere
    args.stat = modeldef
    args.shapes = modeldef['shapes']
    train(args)

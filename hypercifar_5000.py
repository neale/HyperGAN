import os
import sys
import time
import argparse
import numpy as np
from glob import glob
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn
from torch import autograd
from torch import optim
from torch.nn import functional as F
import torch.distributions.multivariate_normal as N
import pprint

import ops
import utils
import netdef
import datagen


def load_args():

    parser = argparse.ArgumentParser(description='param-wgan')
    parser.add_argument('-z', '--z', default=256, type=int, help='latent space width')
    parser.add_argument('-ze', '--ze', default=512, type=int, help='encoder dimension')
    parser.add_argument('-g', '--gp', default=10, type=int, help='gradient penalty')
    parser.add_argument('-b', '--batch_size', default=32, type=int)
    parser.add_argument('-e', '--epochs', default=200000, type=int)
    parser.add_argument('-s', '--model', default='mednet', type=str)
    parser.add_argument('-d', '--dataset', default='cifar', type=str)
    parser.add_argument('--beta', default=100., type=float)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--use_x', default=False, type=bool, help='sample from real layers')
    parser.add_argument('--pretrain_e', default=False, type=bool)
    parser.add_argument('--scratch', default=False, type=bool)
    parser.add_argument('--exp', default='0', type=str)
    parser.add_argument('--hidden', default=False, type=bool)
    args = parser.parse_args()
    return args


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'Encoder_lc'
        self.linear1 = nn.Linear(self.ze, 512)
        self.linear2 = nn.Linear(512, self.z*5)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        #print ('E in: ', x.shape)
        x = x.view(-1, self.ze) #flatten filter size
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        # x = self.relu(self.linear3(x))
        x = x.view(-1, 5, self.z)
        # print (x.shape)
        w1 = x[:, 0]
        w2 = x[:, 1]
        w3 = x[:, 2]
        w4 = x[:, 3]
        w5 = x[:, 4]
        #print ('E out: ', x.shape)
        return w1, w2, w3, w4, w5


""" Convolutional (3 x 16 x 3 x 3) """
class GeneratorW1(nn.Module):
    def __init__(self, args):
        super(GeneratorW1, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW1'
        self.linear1 = nn.Linear(self.z, 256)
        self.linear2 = nn.Linear(256, 432)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        # print ('W1 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        #x = self.linear4(x)
        x = x.view(-1, 16, 3, 3, 3)
        #print ('W1 out: ', x.shape)
        return x


""" Convolutional (32 x 16 x 3 x 3) """
class GeneratorW2(nn.Module):
    def __init__(self, args):
        super(GeneratorW2, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW2'
        self.linear1 = nn.Linear(self.z, 1024)
        self.linear2 = nn.Linear(1024, 4608)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        # print ('W2 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 32, 16, 3, 3)
        # print ('W2 out: ', x.shape)
        return x


""" Convolutional (32 x 32 x 3 x 3) """
class GeneratorW3(nn.Module):
    def __init__(self, args):
        super(GeneratorW3, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW3'
        self.linear1 = nn.Linear(self.z, 1024)
        self.linear2 = nn.Linear(1024, 9216)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        #print ('W3 in : ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 32, 32, 3, 3)
        #print ('W3 out: ', x.shape)
        return x


""" Linear (128 x 64) """
class GeneratorW4(nn.Module):
    def __init__(self, args):
        super(GeneratorW4, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW4'
        self.linear1 = nn.Linear(self.z, 1024)
        self.linear2 = nn.Linear(1024, 8192)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        #print ('W4 in : ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 64, 128)
        #print ('W4 out: ', x.shape)
        return x


""" Linear (64 x 10) """
class GeneratorW5(nn.Module):
    def __init__(self, args):
        super(GeneratorW5, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW5'
        self.linear1 = nn.Linear(self.z, 640)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        #print ('W3 in : ', x.shape)
        x = self.linear1(x)
        x = x.view(-1, 10, 64)
        #print ('W3 out: ', x.shape)
        return x


class DiscriminatorZ(nn.Module):
    def __init__(self, args):
        super(DiscriminatorZ, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        
        self.name = 'Discriminator_z'
        self.linear1 = nn.Linear(self.z, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 1)
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


def sample_x(args, gen, id, grad=True):
    if type(gen) is list:
        res = []
        for i, g in enumerate(gen):
            data = next(g)
            x = (torch.tensor(data, requires_grad=grad)).cuda()
            res.append(x.view(*args.shapes[i]))
    else:
        data = next(gen)
        x = torch.tensor(data, requires_grad=grad).cuda()
        res = x.view(*args.shapes[id])
    return res


def sample_z(args, grad=True):
    z = torch.randn(args.batch_size, args.dim, requires_grad=grad).cuda()
    return z


def sample_z_like(shape, scale=1., grad=True):
    mean = torch.zeros(shape[1])
    cov = torch.eye(shape[1])
    D = N.MultivariateNormal(mean, cov)
    z = D.sample((shape[0],)).cuda()
    return scale * z
 

# hard code the two layer net
def train_clf(args, Z, data, target):
    """ calc classifier loss """
    data, target = data.cuda(), target.cuda()
    x = F.relu(F.conv2d(data, Z[0]))
    x = F.max_pool2d(x, 2, 2)
    x = F.relu(F.conv2d(x, Z[1]))
    x = F.max_pool2d(x, 2, 2)
    x = F.relu(F.conv2d(x, Z[2]))
    x = F.max_pool2d(x, 2, 2)
    x = x.view(x.size(0), -1)
    x = F.relu(F.linear(x, Z[3]))
    x = F.linear(x, Z[4])
    loss = F.cross_entropy(x, target)
    pred = x.data.max(1, keepdim=True)[1]
    correct = pred.eq(target.data.view_as(pred)).long().cpu().sum()
    return (correct, loss)


def batch_zero_grad(nets):
    for module in nets:
        module.zero_grad()


def batch_update_optim(optimizers):
    for optim in optimizers:
        optim.step()

def free_params(nets):
    for module in nets:
        for p in module.parameters():
            p.requires_grad = True


def frozen_params(nets):
    for module in nets:
        for p in module.parameters():
            p.requires_grad = False


def pretrain_loss(encoded, noise):
    mean_z = torch.mean(noise, dim=0, keepdim=True)
    mean_e = torch.mean(encoded, dim=0, keepdim=True)
    mean_loss = F.mse_loss(mean_z, mean_e)

    cov_z = torch.matmul((noise-mean_z).transpose(0, 1), noise-mean_z)
    cov_z /= 999
    cov_e = torch.matmul((encoded-mean_e).transpose(0, 1), encoded-mean_e)
    cov_e /= 999
    cov_loss = F.mse_loss(cov_z, cov_e)
    return mean_loss, cov_loss


def train(args):
    
    torch.manual_seed(8734)
    
    netE = Encoder(args).cuda()
    W1 = GeneratorW1(args).cuda()
    W2 = GeneratorW2(args).cuda()
    W3 = GeneratorW3(args).cuda()
    W4 = GeneratorW4(args).cuda()
    W5 = GeneratorW5(args).cuda()
    netD = DiscriminatorZ(args).cuda()
    print (netE, W1, W2, W3, W4, W5, netD)

    optimE = optim.Adam(netE.parameters(), lr=5e-3, betas=(0.5, 0.9), weight_decay=1e-4)
    optimW1 = optim.Adam(W1.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimW2 = optim.Adam(W2.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimW3 = optim.Adam(W3.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimW4 = optim.Adam(W4.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimW5 = optim.Adam(W5.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimD = optim.Adam(netD.parameters(), lr=5e-5, betas=(0.5, 0.9), weight_decay=1e-4)
    
    best_test_acc, best_test_loss = 0., np.inf
    args.best_loss, args.best_acc = best_test_loss, best_test_acc
    
    if args.hidden:
        c_idx = [0, 1, 2, 3, 4]
        cifar_train, cifar_test = datagen.load_cifar_hidden(args, c_idx)
    else:
        cifar_train, cifar_test = datagen.load_cifar(args)
    one = torch.FloatTensor([1]).cuda()
    mone = (one * -1).cuda()
    print ("==> pretraining encoder")
    j = 0
    final = 100.
    e_batch_size = 1000
    if args.pretrain_e:
        for j in range(2000):
            x = sample_z_like((e_batch_size, args.ze))
            z = sample_z_like((e_batch_size, args.z))
            codes = netE(x)
            for i, code in enumerate(codes):
                code = code.view(e_batch_size, args.z)
                mean_loss, cov_loss = pretrain_loss(code, z)
                loss = mean_loss + cov_loss
                loss.backward(retain_graph=True)
            optimE.step()
            netE.zero_grad()
            print ('Pretrain Enc iter: {}, Mean Loss: {}, Cov Loss: {}'.format(
                j, mean_loss.item(), cov_loss.item()))
            final = loss.item()
            if loss.item() < 0.1:
                print ('Finished Pretraining Encoder')
                break

    print ('==> Begin Training')
    for _ in range(args.epochs):
        for batch_idx, (data, target) in enumerate(cifar_train):
            if batch_idx > 158:
                continue
            batch_zero_grad([netE, W1, W2, W3, W4, W5, netD])
            z = sample_z_like((args.batch_size, args.ze,))
            codes = netE(z)
            l1 = W1(codes[0])
            l2 = W2(codes[1])
            l3 = W3(codes[2])
            l4 = W4(codes[3])
            l5 = W5(codes[4])
            
            # Z Adversary 
            free_params([netD])
            frozen_params([netE, W1, W2, W3, W4, W5])
            for code in codes:
                noise = sample_z_like((args.batch_size, args.z))
                d_real = netD(noise)
                d_fake = netD(code)
                d_real_loss = -1 * torch.log((1-d_real).mean())
                d_fake_loss = -1 * torch.log(d_fake.mean())
                d_real_loss.backward(retain_graph=True)
                d_fake_loss.backward(retain_graph=True)
                d_loss = d_real_loss + d_fake_loss
            optimD.step()

            # Generator (Mean test)
            frozen_params([netD])
            free_params([netE, W1, W2, W3, W4, W5])
            for (g1, g2, g3, g4, g5) in zip(l1, l2, l3, l4, l5):
                correct, loss = train_clf(args, [g1, g2, g3, g4, g5], data, target)
                scaled_loss = args.beta * loss
                scaled_loss.backward(retain_graph=True)
               
            optimE.step(); optimW1.step(); optimW2.step()
            optimW3.step(); optimW4.step(); optimW5.step()
            loss = loss.item()
            
            """ Update Statistics """
            if batch_idx % 50 == 0:
                acc = (correct / 1) 
                norm_z1 = np.linalg.norm(l1.data)
                norm_z2 = np.linalg.norm(l2.data)
                norm_z3 = np.linalg.norm(l3.data)
                norm_z4 = np.linalg.norm(l4.data)
                norm_z5 = np.linalg.norm(l5.data)
                print ('**************************************')
                print ('Mean Test: Enc, Dz, Lscale: {} test'.format(args.beta))
                print ('Acc: {}, G Loss: {}, D Loss: {}'.format(acc, loss, d_loss))
                print ('Filter norm: ', norm_z1)
                print ('Filter norm: ', norm_z2)
                print ('Filter norm: ', norm_z3)
                print ('Linear norm: ', norm_z4)
                print ('Linear norm: ', norm_z5)
                print ('best test loss: {}'.format(args.best_loss))
                print ('best test acc: {}'.format(args.best_acc))
                print ('**************************************')
            if batch_idx % 100 == 0:
                test_acc = 0.
                test_loss = 0.
                for i, (data, y) in enumerate(cifar_test):
                    z = sample_z_like((args.batch_size, args.ze,))
                    w1_code, w2_code, w3_code, w4_code, w5_code = netE(z)
                    l1 = W1(w1_code)
                    l2 = W2(w2_code)
                    l3 = W3(w3_code)
                    l4 = W4(w4_code)
                    l5 = W5(w5_code)
                    for (g1, g2, g3, g4, g5) in zip(l1, l2, l3, l4, l5):
                        correct, loss = train_clf(args, [g1, g2, g3, g4, g5], data, y)
                        test_acc += correct.item()
                        test_loss += loss.item()
                test_loss /= len(cifar_test.dataset) * args.batch_size
                test_acc /= len(cifar_test.dataset) * args.batch_size
                print ('Test Accuracy: {}, Test Loss: {}'.format(test_acc, test_loss))
                if test_loss < best_test_loss or test_acc > best_test_acc:
                    utils.save_hypernet_cifar(args, [netE, W1, W2, W3, W4, W5, netD], test_acc)
                    print ('==> new best stats, saving')
                    if test_loss < best_test_loss:
                        best_test_loss = test_loss
                        args.best_loss = test_loss
                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
                        args.best_acc = test_acc


if __name__ == '__main__':

    args = load_args()
    modeldef = netdef.nets()[args.model]
    pprint.pprint (modeldef)
    # log some of the netstat quantities so we don't subscript everywhere
    args.stat = modeldef
    args.shapes = modeldef['shapes']
    train(args)

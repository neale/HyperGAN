import os
import sys
import time
import argparse
import numpy as np
from glob import glob
from scipy.misc import imshow
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn
from torch import autograd
from torch import optim
from torch.nn import functional as F
import torch.distributions.multivariate_normal as N
import pprint

import utils
import netdef
import datagen
import matplotlib.pyplot as plt


def load_args():

    parser = argparse.ArgumentParser(description='param-wgan')
    parser.add_argument('--z', default=128, type=int, help='latent space width')
    parser.add_argument('--ze', default=300, type=int, help='encoder dimension')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=200000, type=int)
    parser.add_argument('--model', default='small2', type=str)
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--layer', default='all', type=str)
    parser.add_argument('--beta', default=1000, type=int)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--gan', default=False, type=bool)
    parser.add_argument('--use_wae', default=True, type=bool)
    parser.add_argument('--val_iters', default=10, type=bool)
    parser.add_argument('--use_x', default=False, type=bool)
    parser.add_argument('--load_e', default=False, type=bool)
    parser.add_argument('--pretrain_e', default=False, type=bool)
    parser.add_argument('--scratch', default=False, type=bool)
    parser.add_argument('--exp', default='0', type=str)
    parser.add_argument('--use_d', default=False, type=str)

    args = parser.parse_args()
    return args


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'Encoder'
        self.linear1 = nn.Linear(self.ze, 300)
        self.linear2 = nn.Linear(300, 300)
        self.linear3 = nn.Linear(300, self.z*3)
        self.bn1 = nn.BatchNorm1d(300)
        self.bn2 = nn.BatchNorm1d(300)
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
        self.linear1 = nn.Linear(self.z, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 512)
        self.linear4 = nn.Linear(512, 800)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        # print ('W1 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.relu(self.bn3(self.linear3(x)))
        x = self.linear4(x)
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
        self.linear1 = nn.Linear(self.z, 256)
        self.linear2 = nn.Linear(256, 1600)
        self.linear3 = nn.Linear(1600, 6400)
        self.linear4 = nn.Linear(6400, 25600)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(1600)
        self.bn3 = nn.BatchNorm1d(6400)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        #print ('W2 in: ', x.shape)
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
        self.name = 'GeneratorLinear'
        self.linear1 = nn.Linear(self.z, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 512*10)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        #print ('W3 in : ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        x = x.view(-1, 10, 512)
        # x = self.linear_out(x)
        #print ('W3 out: ', x.shape)
        return x


class DiscriminatorZ(nn.Module):
    def __init__(self, args):
        super(DiscriminatorZ, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        
        self.name = 'DiscriminatorZ'
        self.linear1 = nn.Linear(self.z, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 1)
        self.relu = nn.LeakyReLU(inplace=True)
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
            x = torch.tensor(data, requires_grad=grad).cuda()
            res.append(x.view(*args.shapes[i]))
    else:
        data = next(gen)
        x = torch.tensor(data, requires_grad=grad).cuda()
        res = x.view(*args.shapes[id])
    return res


def sample_z(args, grad=True):
    z = torch.randn(args.batch_size, args.dim, requires_grad=grad).cuda()
    return z


def create_d(shape, scale=.1, grad=True):
    mean = torch.zeros(shape)
    cov = torch.eye(shape)
    D = N.MultivariateNormal(mean, cov)
    return D


def sample_d(D, shape, scale=1., grad=True):
    z = scale * D.sample((shape,)).cuda()
    z.requires_grad = grad
    return scale * z


def sample_z_like(shape, scale=1., grad=True):
    return torch.randn(*shape, requires_grad=grad).cuda()


def load_mnist():
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True}
    path = 'data/'
    if args.scratch:
        path = '/scratch/eecs-share/ratzlafn/' + path
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                batch_size=32, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=32, shuffle=True, **kwargs)
    return train_loader, test_loader


# hard code the two layer net
def train_clf(args, Z, data, target, val=False):
    """ calc classifier loss """
    data, target = data.cuda(), target.cuda()
    out = F.conv2d(data, Z[0], stride=1)
    out = F.leaky_relu(out)
    out = F.max_pool2d(out, 2, 2)
    out = F.conv2d(out, Z[1], stride=1)
    out = F.leaky_relu(out)
    out = F.max_pool2d(out, 2, 2)
    out = out.view(-1, 512)
    out = F.linear(out, Z[2])
    loss = F.cross_entropy(out, target)
    correct = None
    if val:
        pred = out.data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).long().cpu().sum()
    return (correct, loss)


def batch_zero_grad(modules):
    for module in modules:
        module.zero_grad()


def batch_update_optim(optimizers):
    for optimizer in optimizers:
        optimizer.step()


def free_params(modules):
    for module in modules:
        for p in module.parameters():
            p.requires_grad = False


def frozen_params(modules):
    for module in modules:
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
    netD = DiscriminatorZ(args).cuda()
    print (netE, W1, W2, W3)

    optimE = optim.Adam(netE.parameters(), lr=.005, betas=(0.5, 0.9), weight_decay=1e-4)
    optimW1 = optim.Adam(W1.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimW2 = optim.Adam(W2.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimW3 = optim.Adam(W3.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimD = optim.Adam(netD.parameters(), lr=1e-6, betas=(0.5, 0.9), weight_decay=1e-4)
    
    best_test_acc, best_test_loss = 0., np.inf
    args.best_loss, args.best_acc = best_test_loss, best_test_acc
    if args.resume:
        stats = (0., np.inf)
        netE, optimE, _ = load_model(args, netE, optimE)
        W1, optimW1, _ = load_model(args, W1, optimW1)
        W2, optimW2, _ = load_model(args, W2, optimW2)
        W3, optimW3, _ = load_model(args, W3, optimW3)
        netD, optimD, _ = load_model(args, netD, optimD)
        best_test_acc, best_test_loss = stats
        print ('==> resumeing models at ', stats)

    mnist_train, mnist_test = load_mnist()
    if args.use_x:
        base_gen = datagen.load(args)
        w1_gen = utils.inf_train_gen(base_gen[0])
        w2_gen = utils.inf_train_gen(base_gen[1])
        w3_gen = utils.inf_train_gen(base_gen[2])
        X = sample_x(args, [w1_gen, w2_gen, w3_gen], 0)
        X = list(map(lambda x: (x+1e-10).float(), X))

    x_dist = create_d(args.ze)
    z_dist = create_d(args.z)
    one = torch.FloatTensor([1]).cuda()
    mone = (one * -1).cuda()
    print ("==> pretraining encoder")
    j = 0
    final = 100.
    e_batch_size = 1000
    if args.load_e:
        netE, optimE, _ = utils.load_model(args, netE, optimE)
        print ('==> loading pretrained encoder')
    if args.pretrain_e:
        for j in range(2000):
            #x = sample_x(args, [w1_gen, w2_gen, w3_gen, w4_gen, w5_gen], 0)
            x = sample_d(x_dist, e_batch_size)
            z = sample_d(z_dist, e_batch_size)
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
        utils.save_model(args, netE, optimE)
    print ('==> Begin Training')
    for _ in range(1000):
        for batch_idx, (data, target) in enumerate(mnist_train):

            batch_zero_grad([netE, W1, W2, W3, netD])
            z = sample_d(x_dist, args.batch_size)
            codes = netE(z)
            l1 = W1(codes[0]).mean(0)
            l2 = W2(codes[1]).mean(0)
            l3 = W3(codes[2]).mean(0)
            
            if args.use_d:
                free_params([netD])
                frozen_params([netE, W1, W2, W3])
                for code in codes:
                    noise = sample_d(z_dist, args.batch_size)
                    d_real = netD(noise)
                    d_fake = netD(code)
                    d_real_loss = -1 * torch.log((1-d_real).mean())
                    d_fake_loss = -1 * torch.log(d_fake.mean())
                    d_real_loss.backward(retain_graph=True)
                    d_fake_loss.backward(retain_graph=True)
                    d_loss = d_real_loss + d_fake_loss
                optimD.step()

                frozen_params([netD])
                free_params([netE, W1, W2, W3])
            
            correct, loss = train_clf(args, [l1, l2, l3], data, target, val=True)
            scaled_loss = args.beta*loss
            scaled_loss.backward()
            optimE.step()
            optimW1.step()
            optimW2.step()
            optimW3.step()
            loss = loss.item()
                
            if batch_idx % 50 == 0:
                acc = (correct / 1) 
                norm_z1 = np.linalg.norm(l1.detach())
                norm_z2 = np.linalg.norm(l2.detach())
                norm_z3 = np.linalg.norm(l3.detach())
                print ('**************************************')
                print ('Mean MNIST test Dz Lscale: {}'.format(args.beta))
                #print ('Acc: {}, Loss: {}, D Loss: {}'.format(acc, loss, d_loss))
                print ('Acc: {}, Loss: {}'.format(acc, loss))
                print ('Filter norm: ', norm_z1, '-->', norm_z1)
                print ('Filter norm: ', norm_z2, '-->', norm_z2)
                print ('Linear norm: ', norm_z3, '-->', norm_z3)
                print ('best test loss: {}'.format(args.best_loss))
                print ('best test acc: {}'.format(args.best_acc))
                print ('**************************************')

            if batch_idx % 100 == 0:
                test_acc = 0.
                test_loss = 0.
                for i, (data, y) in enumerate(mnist_test):
                    z = sample_d(x_dist, args.batch_size)
                    codes = netE(z)
                    l1 = W1(codes[0]).mean(0)
                    l2 = W2(codes[1]).mean(0)
                    l3 = W3(codes[2]).mean(0)
                    min_loss_batch = 10.
                    z_test = [l1, l2, l3]
                    correct, loss = train_clf(args, z_test, data, y, val=True)
                    test_acc += correct.item()
                    test_loss += loss.item()
                test_loss /= len(mnist_test.dataset)
                test_acc /= len(mnist_test.dataset)
                print ('Test Accuracy: {}, Test Loss: {}'.format(test_acc, test_loss))
                if test_loss < best_test_loss or test_acc > best_test_acc:
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

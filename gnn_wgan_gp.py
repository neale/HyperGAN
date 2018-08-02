import os
import sys
import time
import argparse
import numpy as np
from glob import glob
from scipy.misc import imshow
from comet_ml import Experiment
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn
from torch import autograd
from torch import optim
from torch.nn import functional as F
import pprint

import ops
import plot
import utils
import netdef
import datagen
import matplotlib.pyplot as plt


def load_args():

    parser = argparse.ArgumentParser(description='param-wgan')
    parser.add_argument('-z', '--dim', default=64, type=int, help='latent space width')
    parser.add_argument('-ze', '--ze', default=256, type=int, help='encoder dimension')
    parser.add_argument('-g', '--gp', default=10, type=int, help='gradient penalty')
    parser.add_argument('-b', '--batch_size', default=32, type=int)
    parser.add_argument('-e', '--epochs', default=200000, type=int)
    parser.add_argument('-s', '--model', default='small2', type=str)
    parser.add_argument('-d', '--dataset', default='mnist', type=str)
    parser.add_argument('-l', '--layer', default='all', type=str)
    parser.add_argument('-zd', '--depth', default=2, type=int, help='latent space depth')
    parser.add_argument('--nfe', default=64, type=int)
    parser.add_argument('--nfgc', default=64, type=int)
    parser.add_argument('--nfgl', default=64, type=int)
    parser.add_argument('--nfd', default=128, type=int)
    parser.add_argument('--beta', default=1000, type=int)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--comet', default=False, type=bool)
    parser.add_argument('--gan', default=False, type=bool)
    parser.add_argument('--use_wae', default=True, type=bool)
    parser.add_argument('--val_iters', default=10, type=bool)

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
        self.linear3 = nn.Linear(300, 3072)
        self.bn1 = nn.BatchNorm2d(300)
        self.bn2 = nn.BatchNorm2d(300)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        #print ('E in: ', x.shape)
        x = x.view(self.batch_size, -1) #flatten filter size
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        # x = self.relu(self.linear3(x))
        x = x.view(-1, 96, 32)
        # print (x.shape)
        w1 = x[:, :32]
        w2 = x[:, 32:64]
        w3 = x[:, 64:]

        #print ('E out: ', x.shape)
        return w1, w2, w3


class GeneratorW1(nn.Module):
    def __init__(self, args):
        super(GeneratorW1, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW1'
        self.linear1 = nn.Linear(32, 50)
        self.linear2 = nn.Linear(50, 50)
        self.linear3 = nn.Linear(50, 25)
        self.bn1 = nn.BatchNorm2d(50)
        self.bn2 = nn.BatchNorm2d(50)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        #print ('W1 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        x = x.view(-1, 1, 5, 5)
        #print ('W1 out: ', x.shape)
        return x


class GeneratorW2(nn.Module):
    def __init__(self, args):
        super(GeneratorW2, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW2'
        self.linear1 = nn.Linear(32, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 800)
        self.bn1 = nn.BatchNorm2d(100)
        self.bn2 = nn.BatchNorm2d(100)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        #print ('W2 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        x = x.view(-1, 32, 5, 5)
        #print ('W2 out: ', x.shape)
        return x


class GeneratorW3(nn.Module):
    def __init__(self, args):
        super(GeneratorW3, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW3'
        self.linear1 = nn.Linear(self.ze*4, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 512*10)
        self.linear_out = nn.Linear(self.ze, 490)
        self.bn1 = nn.BatchNorm2d(100)
        self.bn2 = nn.BatchNorm2d(100)
        self.bn3 = nn.BatchNorm2d(400)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        #print ('W3 in : ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        x = x.view(-1, 10, 512)
        # x = self.linear_out(x)
        #print ('W3 out: ', x.shape)
        return x


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        
        self.name = 'Discriminator'
        self.pp1 = nn.Linear(800, 256)
        self.pp2 = nn.Linear(160, 128)
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, 25)
        self.linear3 = nn.Linear(25, 25)
        self.linear4 = nn.Linear(25, 1)
        self.relu = nn.LeakyReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, id):
        #print ('Dz in: ', x.shape)
        x = x.view(self.batch_size, -1)
        if id != 0:
            if id == 1:
                x = self.relu(self.pp1(x))
                x = self.relu(self.linear1(x))
            if id == 2:
                x = self.relu(self.pp2(x))
            x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        #print ('Dz out: ', x.shape)
        return x


def sample_x(args, gen, id):
    if type(gen) is list:
        res = []
        for i, g in enumerate(gen):
            data = next(g)
            x = autograd.Variable(torch.Tensor(data)).cuda()
            res.append(x.view(*args.shapes[i]))
    else:
        data = next(gen)
        x = torch.Tensor(data).cuda()
        x = x.view(*args.shapes[id])
        res = autograd.Variable(x)
    return res


def sample_z(args):
    z = torch.randn(args.batch_size, args.dim).cuda()
    z = autograd.Variable(z)
    return z


def sample_z_like(shape):
    z = torch.rand(*shape).cuda()
    z = autograd.Variable(z)
    return z
 

def train_ae(args, netG, netE, x):
    netG.zero_grad()
    netE.zero_grad()
    x_encoding = netE(x)
    x_fake = ops.gen_layer(args, netG, x_encoding)
    # fake = netG(encoding)
    x_fake = x_fake.view(*args.shapes[args.id])
    ae_loss = F.mse_loss(x_fake, x)
    return ae_loss, x_fake


def train_wadv(args, netDz, netE, x, z):
    netDz.zero_grad()
    z_fake = netE(x).view(args.batch_size, -1)
    Dz_real = netDz(z)
    Dz_fake = netDz(z_fake)
    Dz_loss = -(torch.mean(Dz_real) - torch.mean(Dz_fake))
    Dz_loss.backward()


def train_adv(args, netD, x, z):
    netD.zero_grad()
    D_real = netD(x).mean()
    z = z.view(*args.shapes[args.id])
    # fake = netG(z)
    D_fake = netD(z).mean()
    gradient_penalty = calc_gradient_penalty(args, netD,
            x.data, z.data)
    return D_real, D_fake, gradient_penalty


def load_mnist():
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                batch_size=64, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=64, shuffle=True, **kwargs)
    return train_loader, test_loader


# hard code the two layer net
def train_clf(args, Z, data, target, val=False):
    """ calc classifier loss """
    data = autograd.Variable(data).cuda(),
    target = autograd.Variable(target).cuda()
    #im = data[0][0, 0, :, :].cpu().data.numpy()
    #import matplotlib.pyplot as plt
    #plt.imshow(im, cmap='gray')
    #plt.show()
    out = F.conv2d(data[0], Z[0], stride=1)
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
    # (clf_acc, clf_loss), (oa, ol) = ops.clf_loss(args, Z)
    return (correct, loss)


def train_gen(args, netG, netD):
    netG.zero_grad()
    z = sample_z(args)
    fake = netG(z)
    G = netD(fake).mean()
    G_cost = -G
    return G


def cov(x, y):
    mean_x = torch.mean(x, dim=0, keepdim=True)
    mean_y = torch.mean(y, dim=0, keepdim=True)
    cov_x = torch.matmul((x-mean_x).transpose(0, 1), x-mean_x)
    cov_x /= 999
    cov_y = torch.matmul((y-mean_y).transpose(0, 1), y-mean_y)
    cov_y /= 999
    cov_loss = F.mse_loss(cov_y, cov_x)
    return cov_loss


def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def freeze_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def gradient_penalty(args, id, netD, real_data, gen_data):
    batch_size = args.batch_size
    datashape = args.shapes[id]
    alpha = torch.rand(datashape[0], 1)
    #alpha = alpha.expand(datashape[0], real_data.nelement()/datashape[0])
    alpha = alpha.expand(datashape[0], int(np.prod(datashape[1:])))
    alpha = alpha.contiguous().view(*datashape).cuda()
    interpolates = alpha * real_data + ((1 - alpha) * gen_data).cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates, id)
    gradients = autograd.grad(outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.gp
    return gradient_penalty


def train(args):
    
    torch.manual_seed(1)
    
    netE = Encoder(args).cuda()
    W1 = GeneratorW1(args).cuda()
    W2 = GeneratorW2(args).cuda()
    W3 = GeneratorW3(args).cuda()
    netD = Discriminator(args).cuda()
    print (netE, W1, W2, W3, netD)

    optimizerE = optim.Adam(netE.parameters(), lr=5e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimizerW1 = optim.Adam(W1.parameters(), lr=5e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimizerW2 = optim.Adam(W2.parameters(), lr=5e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimizerW3 = optim.Adam(W3.parameters(), lr=5e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimizerD = optim.Adam(netD.parameters(), lr=1e-3, betas=(0.5, 0.9), weight_decay=1e-4)
    
    mnist_train, mnist_test = load_mnist()
    base_gen = datagen.load(args)
    w1_gen = utils.inf_train_gen(base_gen[0])
    w2_gen = utils.inf_train_gen(base_gen[1])
    w3_gen = utils.inf_train_gen(base_gen[2])

    one = torch.FloatTensor([1]).cuda()
    mone = (one * -1).cuda()
    best_acc, best_loss = 0., 1000. 
    for _ in range(1000):
        for batch_idx, (data, target) in enumerate(mnist_train):

            # Discriminator
            free_params(netD)
            netD.zero_grad()
            w1_code = sample_z_like((args.batch_size, 32, 32))
            w2_code = sample_z_like((args.batch_size, 32, 32))
            w3_code = sample_z_like((args.batch_size, 32, 32))
            w1_out, w2_out = [], []
            for i in range(32):
                w1_out.append(W1(w1_code[i]))
                w2_out.append(W2(w2_code[i]))
            l1 = torch.stack(w1_out)
            l2 = torch.stack(w2_out)
            l3 = W3(w3_code.contiguous().view(args.batch_size, -1))
            
            for (z1, z2, z3) in zip(l1, l2, l3):
                #print (z1.shape, z2.shape, z3.shape)
                x1, x2, x3 = sample_x(args, [w1_gen, w2_gen, w3_gen], 0)
                d_real = netD(x1,0).mean() + netD(x2,1).mean() + netD(x3,2).mean()
                d_real.backward(mone)
                dz1, dz2, dz3 = netD(z1,0).mean(), netD(z2,1).mean(), netD(z3,2).mean()
                d_fake = dz1 + dz2 + dz3
                d_fake.backward(one, retain_graph=True)
                #print (z1.shape, x1.shape)
                gp1 = gradient_penalty(args, 0, netD, z1.data, x1.data)
                gp2 = gradient_penalty(args, 1, netD, z2.data, x2.data)
                gp3 = gradient_penalty(args, 2, netD, z3.data, x3.data)
                total_gp = gp1 + gp2 + gp3
                total_gp.backward()
                W1_dist = d_real - d_fake
            optimizerD.step()

            # Weight generators
            W1.zero_grad()
            W2.zero_grad()
            W3.zero_grad()
            freeze_params(netD)
            w1_code = sample_z_like((args.batch_size, 32, 32))
            w2_code = sample_z_like((args.batch_size, 32, 32))
            w3_code = sample_z_like((args.batch_size, 32, 32))
            w1_out, w2_out = [], []
            for i in range(32):
                w1_out.append(W1(w1_code[i]))
                w2_out.append(W2(w2_code[i]))
            l1 = torch.stack(w1_out)
            l2 = torch.stack(w2_out)
            l3 = W3(w3_code.contiguous().view(args.batch_size, -1))

            for (z1, z2, z3) in zip(l1, l2, l3):
                correct, loss = train_clf(args, [z1, z2, z3], data, target, val=True)
                (acc, loss), _ = utils.test_samples(args, [z1, z2, z3], train=True)
                scaled_loss = (100*loss) #+ z1_loss + z2_loss + z3_loss
                scaled_loss.backward(retain_graph=True)
                d1 = netD(z1,0).mean() + netD(z2,1).mean() + netD(z3,2).mean()
                d1.backward(mone, retain_graph=True)
            optimizerW1.step()
            optimizerW2.step()
            optimizerW3.step()
            loss = loss.cpu().data.numpy()[0]

            if batch_idx % 50 == 0:
                acc = correct 
                norm_x, norm_x2 = np.linalg.norm(x1.data), np.linalg.norm(x2.data)
                norm_z1, norm_z2 = np.linalg.norm(z1.data), np.linalg.norm(z2.data)
                print ('*****************************')
                print ('Acc: ', acc, 'Loss: ', loss)
                print ('Filter norm: ', norm_z1, '-->', norm_x)
                print ('Linear norm: ', norm_z2, '-->', norm_x2)
                print ('W1 dist: ', W1_dist.cpu().data.numpy()[0])
                print ('GP: ', total_gp.cpu().data.numpy()[0])
                print 
                print ('*****************************')
            if batch_idx % 100 == 0:
                acc, zacc = 0., 0.
                test_loss, ztest_loss = 0., 0.
                for i, (data, y) in enumerate(mnist_test):
                    # x, x2 = sample_x(args, [conv_gen, linear_gen], 0)
                    w1_code = sample_z_like((args.batch_size, 32, 32))
                    w2_code = sample_z_like((args.batch_size, 32, 32))
                    w3_code = sample_z_like((args.batch_size, 32, 32))
                    w1_out, w2_out = [], []
                    for j in range(32):
                        w1_out.append(W1(w1_code[j]))
                        w2_out.append(W2(w2_code[j]))
                    l1 = torch.stack(w1_out)
                    l2 = torch.stack(w2_out)
                    l3 = W3(w3_code.contiguous().view(args.batch_size, -1))
                    for (z1, z2, z3) in zip(l1, l2, l3):
                        correct, loss = train_clf(args, [z1, z2, z3], data, y, val=True)
                        acc += correct
                        test_loss += (loss.cpu().data.numpy()[0])
                test_loss /= len(mnist_test.dataset) * 32
                acc /= len(mnist_test.dataset) * 32
                print ("test loss: {}, acc: {} ".format(test_loss, acc))
                if acc > best_acc or (acc == best_acc and loss < best_loss):
                    utils.save_model(args, W1, optimizerW1)
                    utils.save_model(args, W2, optimizerW2)
                    utils.save_model(args, W3, optimizerW3)
                    utils.save_model(args, netD, optimizerD)

            
if __name__ == '__main__':

    args = load_args()
    modeldef = netdef.nets()[args.model]
    pprint.pprint (modeldef)
    # log some of the netstat quantities so we don't subscript everywhere
    args.stat = modeldef
    args.shapes = modeldef['shapes']
    args.lcd = modeldef['base_shape']
    # why is a running product so hard in python
    args.gcd = int(np.prod([*args.shapes[0]]))
    train(args)

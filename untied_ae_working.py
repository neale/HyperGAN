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
    parser.add_argument('-g', '--gp', default=10, type=int, help='gradient penalty')
    parser.add_argument('-b', '--batch_size', default=64, type=int)
    parser.add_argument('-e', '--epochs', default=200000, type=int)
    parser.add_argument('-s', '--model', default='small', type=str)
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
        self.linear0 = nn.Linear(self.lcd*self.lcd*10, self.lcd*self.lcd)
        self.linear1 = nn.Linear(self.lcd*self.lcd, self.nfe*4)
        self.linear2 = nn.Linear(self.nfe*4, self.nfe*2)
        self.linear3 = nn.Linear(self.nfe*2, self.dim)
        self.bn0 = nn.BatchNorm2d(self.lcd*self.lcd)
        self.bn1 = nn.BatchNorm2d(self.nfe*4)
        self.bn2 = nn.BatchNorm2d(self.nfe*2)
        self.bn3 = nn.BatchNorm2d(self.dim)
        self.relu = nn.LeakyReLU(.2, inplace=True)

    def forward(self, x):
        #print ('E in: ', x.shape)
        x = x.view(self.batch_size, -1) #flatten filter size
        if x.shape[-1] > self.lcd*self.lcd:
            x = self.relu(self.linear0(x))
        if self.use_wae:
            z = torch.normal(torch.zeros_like(x.data), std=0.01)
            x.data += z
        
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        # print ('E out: ', x.shape)
        return x


class GeneratorConv(nn.Module):
    def __init__(self, args):
        super(GeneratorConv, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorConv'
        self.linear1 = nn.Linear(self.dim, self.nfgc*4)
        self.linear2 = nn.Linear(self.nfgc*4, self.nfgc*2)
        self.linear3 = nn.Linear(self.nfgc*2, self.nfgc)
        self.linear_out = nn.Linear(self.nfgc, self.lcd*self.lcd)
        self.relu = nn.LeakyReLU(.2, inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        #print ('Gc in: ', x.shape)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        z = torch.normal(torch.zeros_like(x.data), std=0.1)
        x.data += z
        x = self.linear_out(x)
        x = x.view(*self.shapes[0])
        #print ('Gc out: ', x.shape)
        return x


class GeneratorLinear(nn.Module):
    def __init__(self, args):
        super(GeneratorLinear, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorLinear'
        self.linear1 = nn.Linear(self.dim, self.nfgl*2)
        self.linear2 = nn.Linear(self.nfgl*2, self.nfgl*4)
        self.linear3 = nn.Linear(self.nfgl*4, self.nfgl*4)
        self.linear_out = nn.Linear(self.nfgl*4, self.lcd*self.lcd*10)
        self.relu = nn.LeakyReLU(.2, inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        #print ('Gl in : ', x.shape)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        z = torch.normal(torch.zeros_like(x.data), std=0.001)
        x.data += z
        x = self.linear_out(x)
        x = x.view(*self.shapes[1])
        #print ('Gl out: ', x.shape)
        return x


class DiscriminatorZ(nn.Module):
    def __init__(self, args):
        super(DiscriminatorZ, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        
        self.name = 'Discriminator_wae'
        self.linear0 = nn.Linear(self.lcd*self.lcd*10, self.lcd*self.lcd)
        self.linear1 = nn.Linear(self.lcd*self.lcd, 256)
        #self.linear1 = nn.Linear(self.dim, 256)
        self.linear2 = nn.Linear(256, 1)
        self.relu = nn.LeakyReLU(.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print ('Dz in: ', x.shape)
        x = x.view(self.batch_size, -1)
        if x.shape[-1] > self.lcd*self.lcd:
            x = self.relu(self.linear0(x))
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.sigmoid(x)
        # print ('Dz out: ', x.shape)
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
    out = F.conv2d(data[0], Z[0], padding=4)
    out = F.elu(out)
    out = F.max_pool2d(out, 4, 4)
    out = out.view(-1, 3136)
    out = F.linear(out, Z[1])
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


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def calc_gradient_penalty(args, netD, real_data, gen_data):
    batch_size = args.batch_size
    datashape = args.shapes[args.id]
    alpha = torch.rand(datashape[0], 1)
    #alpha = alpha.expand(datashape[0], real_data.nelement()/datashape[0])
    alpha = alpha.expand(datashape[0], int(np.prod(datashape[1:])))
    alpha = alpha.contiguous().view(*datashape).cuda()
    interpolates = alpha * real_data + ((1 - alpha) * gen_data).cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)
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
    netGc = GeneratorConv(args).cuda()
    netGl = GeneratorLinear(args).cuda()
    netD = DiscriminatorZ(args).cuda()
    print (netE, netGc, netGl, netD)

    optimizerE = optim.Adam(netE.parameters(), lr=1e-3, betas=(0.5, 0.9), weight_decay=1e-4)
    optimizerGc = optim.Adam(netGl.parameters(), lr=1e-3, betas=(0.5, 0.9), weight_decay=1e-4)
    optimizerGl = optim.Adam(netGc.parameters(), lr=1e-3, betas=(0.5, 0.9), weight_decay=1e-4)
    optimizerD = optim.Adam(netD.parameters(), lr=1e-3, betas=(0.5, 0.9), weight_decay=1e-4)
    
    mnist_train, mnist_test = load_mnist()
    base_gen = datagen.load(args)
    conv_gen = utils.inf_train_gen(base_gen[0])
    linear_gen = utils.inf_train_gen(base_gen[1])

    one = torch.FloatTensor([1]).cuda()
    mone = (one * -1).cuda()
    """
    print ('==> running clf experiments')
    for _ in range(40):
        acc = 0
        for batch_idx, (data, target) in enumerate(mnist_test):
            conv = autograd.Variable(torch.Tensor(next(conv_gen))).cuda()
            linear = autograd.Variable(torch.Tensor(next(linear_gen))).cuda()
            # add noise
            z_c = torch.rand(conv.data.shape).cuda()
            #conv.data += z_c
            z_l = torch.rand(linear.data.shape).cuda()/10.
            linear.data += z_l
            c, l = train_clf(args, [conv, linear], data, target, val=True)
            acc += c
        print (np.linalg.norm(linear.data))
        print (float(acc) / len(mnist_test.dataset) * 100, '%')
    sys.exit(0)
    """
    for _ in range(1000):
        for batch_idx, (data, target) in enumerate(mnist_train):

            netE.zero_grad()
            netGc.zero_grad()
            netGl.zero_grad()
            
            x, x2 = sample_x(args, [conv_gen, linear_gen], 0)
            t1 = netGc(netE(x))
            t1_loss = F.mse_loss(t1, x)
            t2 = netGl(netE(x2))
            t2_loss = F.mse_loss(t2, x2)
            
            correct, loss = train_clf(args, [t1, t2], data, target, val=True)
            scaled_loss = (10*loss) + t2_loss + t1_loss
            scaled_loss.backward()
            optimizerE.step()
            optimizerGc.step()
            optimizerGl.step()
            loss = loss.cpu().data.numpy()[0]
            
            if args.gan:
                for i in range(1):
                    netD.zero_grad()
                    x1, x2 = sample_x(args, [conv_gen, linear_gen], 0)
                    d_real = netD(x1).mean()
                    d_real.backward(mone)
                    d_real = netD(x2).mean()
                    d_real.backward(mone)
                    args.id = 0
                    z1 = autograd.Variable(torch.randn(args.batch_size, args.dim)).cuda()
                    g_fake = netGc(z1)
                    d_fake = netD(g_fake).mean()
                    d_fake.backward(one)
                    gp = calc_gradient_penalty(args, netD, g_fake.data, x1.data)
                    gp.backward()
                    args.id = 1
                    z2 = autograd.Variable(torch.randn(args.batch_size, args.dim)).cuda()
                    g_fake = netGl(z2)
                    d_fake = netD(g_fake).mean()
                    d_fake.backward(one)
                    gp = calc_gradient_penalty(args, netD, g_fake.data, x2.data)
                    gp.backward()
                    
                    optimizerD.step()
        
                z = sample_z(args)
                fake_c = netGc(z)
                fake_l = netGl(z)
                G_c = netD(fake_c).mean()
                G_c.backward(mone)
                optimizerGc.step()
                G_l = netD(fake_l).mean()
                G_l.backward(mone)
                optimizerGl.step()

            if batch_idx % 50 == 0:
                acc = (correct / 1) 
                norm_x = np.linalg.norm(x.data)
                norm_x2 = np.linalg.norm(x2.data)
                norm_t1 = np.linalg.norm(t1.data)
                norm_t2 = np.linalg.norm(t2.data)
                print (acc, loss, 'CONV-- G: ', norm_t1, '-->', norm_x,
                        'LINEAR-- G: ', norm_t2, '-->', norm_x2)
                        #'Dz -- ', d_loss1.cpu().data[0], d_loss2.cpu().data[0])
            if batch_idx % 500 == 0:
                acc, zacc = 0., 0.
                test_loss, ztest_loss = 0., 0.
                for i, (data, y) in enumerate(mnist_test):
                    x, x2 = sample_x(args, [conv_gen, linear_gen], 0)
                    t1 = netE(x)
                    t1_fake = netGc(t1)
                    t1_loss = F.mse_loss(t1_fake, x)
                    t2 = netE(x2)
                    t2_fake = netGl(t2)
                    t2_loss = F.mse_loss(t2_fake, x2)
                    correct, loss = train_clf(args, [t1_fake, t2_fake], data, y, val=True)
                    acc += correct
                    test_loss += loss.cpu().data.numpy()[0]
                test_loss /= len(mnist_test.dataset)
                acc /= len(mnist_test.dataset)
                if args.gan:
                    for i, (data, y) in enumerate(mnist_test):
                        z = sample_z(args)
                        t1_fake = netGc(z)
                        t2_fake = netGl(z)
                        correct, loss = train_clf(args, [t1_fake, t2_fake], data, y, val=True)
                        zacc += correct
                        ztest_loss += loss.cpu().data.numpy()[0]
                    ztest_loss /= len(mnist_test.dataset)
                    zacc /= len(mnist_test.dataset)

                    print ("test loss: {}, acc: {} ||| z test loss: {}, z acc {}".format(test_loss, acc, ztest_loss, zacc))
                else:
                    print ("test loss: {}, acc: {} ".format(test_loss, acc))

                """
                plt.ion()
                fig, (ax, ax1) = plt.subplots(1, 2)
                x = [x.cpu().data.numpy().flatten(), z1.cpu().data.numpy().flatten()]
                for i in range(len(x)):
                    n, bins, patches = ax.hist(x[i], 50, density=True, alpha=0.75, label=str(i))
                ax.legend(loc='upper right')
                ax.set_title('conv1')
                ax.grid(True)
                y = [x2.cpu().data.numpy().flatten(), z2.cpu().data.numpy().flatten()]
                for i in range(len(y)):
                    n, bins, patches = ax1.hist(y[i], 50, density=True, alpha=0.75, label=str(i))
                ax1.legend(loc='upper right')
                ax1.set_title('linear')
                ax1.grid(True)

                plt.draw()
                plt.pause(1.0)
                plt.close()
                """
            
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

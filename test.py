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
from torch import nn
from torch import autograd
from torch import optim
from torch.nn import functional as F
import pprint
import ops
import plot
import utils
import netdef

def load_args():

    parser = argparse.ArgumentParser(description='param-wgan')
    parser.add_argument('-z', '--dim', default=64, type=int, help='latent space size')
    parser.add_argument('-g', '--gp', default=10, type=int, help='gradient penalty')
    parser.add_argument('-b', '--batch_size', default=64, type=int)
    parser.add_argument('-e', '--epochs', default=200000, type=int)
    parser.add_argument('-s', '--model', default='small', type=str)
    parser.add_argument('-d', '--dataset', default='mnist', type=str)
    parser.add_argument('-l', '--layer', default='all', type=str)
    parser.add_argument('--nfe', default=64, type=int)
    parser.add_argument('--nfg', default=64, type=int)
    parser.add_argument('--nfd', default=128, type=int)
    parser.add_argument('--beta', default=10, type=float)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--comet', default=False, type=bool)
    parser.add_argument('--use_wae', default=False, type=bool)

    args = parser.parse_args()
    return args


class Encoder_fc(nn.Module):
    def __init__(self, args):
        super(Encoder_fc, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'Encoder'
        self.linear1 = nn.Linear(self.lcd*self.lcd, self.nfe*4)
        self.linear2 = nn.Linear(self.nfe*4, self.nfe*2)
        self.linear3 = nn.Linear(self.nfe*2, self.dim)
        self.relu = nn.LeakyReLU(.2, inplace=True)

    def forward(self, x):
        print ('E in: ', x.shape)
        x = x.view(-1, self.lcd*self.lcd) #flatten filter size
        if self.use_wae and self.is_training:
            z = torch.normal(torch.zeros_like(x.data), std=0.01)
            x.data += z
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        print ('E out: ', x.shape)
        return x


class Generator_fc(nn.Module):
    def __init__(self, args):
        super(Generator_fc, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'Generator'
        self.linear1 = nn.Linear(self.dim, self.nfg*4)
        self.linear2 = nn.Linear(self.nfg*4, self.nfg*2)
        self.linear3 = nn.Linear(self.nfg*2, self.nfg)
        self.linear_out = nn.Linear(self.nfg, self.lcd*self.lcd)
        self.relu = nn.LeakyReLU(.2, inplace=True)

    def forward(self, x):
        # print ('G in: ', x.shape)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear_out(x)
        x = x.view(-1, self.lcd, self.lcd)
        # print ('G out: ', x.shape)
        return x


class Generator_conv(nn.Module):
    def __init__(self, args):
        super(Generator_conv, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)

        self.name = 'Generator'
        self.linear = nn.Linear(self.nfg, 64*64)
        self.conv1 = nn.Conv2d(1, self.nfg, 3, 2)
        self.conv2 = nn.Conv2d(self.nfg, self.nfg*2, 3, 2)
        self.conv3 = nn.Conv2d(self.nfg*2, self.nfg*4, 3, 2)
        self.relu = nn.LeakyReLU(.2, inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        print ('G in: ', x.shape)
        x = self.elu(self.linear(x))
        x = x.view(-1, 1, 64, 64)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = x.view(-1, 128, 7, 7)
        print ('G out: ', x.shape)
        return x



class Discriminator_fc(nn.Module):
    def __init__(self, args):
        super(Discriminator_fc, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)

        self.name = 'Discriminator'
        self.ng = self.lcd * self.lcd
        self.linear1 = nn.Linear(self.ng, self.nfd)
        self.linear2 = nn.Linear(self.nfd, self.nfd)
        self.linear3 = nn.Linear(self.nfd, 1)
        self.relu = nn.LeakyReLU(.2, inplace=True)

    def forward(self, x):
        print ('D in: ', x.shape)
        x = x.view(-1, self.ng)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        print ('D out: ', x.shape)
        return x

class Discriminator_z_fc(nn.Module):
    def __init__(self, args, datashape):
        super(Discriminator_z_fc, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        
        self.name = 'Discriminator'
        self.linear1 = nn.Linear(dim, nf)
        self.linear2 = nn.Linear(nf, 1)
        self.relu = nn.LeakyReLU(.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print ('Dz in: ', x.shape)
        # x = x.view(-1, self.ng)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.sigmoid(x)
        # print ('Dz out: ', x.shape)
        return x


def calc_gradient_penalty(args, netD, real_data, gen_data):
    batch_size = args.batch_size
    datashape = args.shapes[args.id]

    alpha = torch.rand(datashape[0], 1)
    alpha = alpha.expand(datashape[0], int(real_data.nelement()/datashape[0]))
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


def sample_x(args, gen, id):
    print ('data {}'.format(id))
    print (gen)
    data = next(gen)
    print ('data {}\n{}'.format(id, data))
    x = torch.Tensor(data).cuda()
    x = x.view(*args.shapes[id])
    x = autograd.Variable(x)
    return x


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
    print ("==> G final out ", x_fake.shape)
    ae_loss = F.mse_loss(x_fake, x)
    return ae_loss


def train_wadv(args, netDz, netE, x, z):
    netDz.zero_grad()
    z_fake = netE(x).view(args.batch_size, -1)
    Dz_real = netDz(z)
    Dz_fake = netDz(z_fake)
    Dz_loss = -(torch.mean(Dz_real) - torch.mean(Dz_fake))
    Dz_loss.backward()


def train_adv(args, netD, netG, x, z):
    netD.zero_grad()
    print ("d real x, ", x.shape)
    D_real = netD(x).mean()
    z = z.view(*args.shapes[args.id])
    # fake = netG(z)
    print (" d fake z ", z.shape)
    D_fake = netD(z).mean()
    D_fake.backward(torch.Tensor([1]).cuda())
    gradient_penalty = calc_gradient_penalty(args, netD,
            x.data, z.data)
    return D_real, D_fake, gradient_penalty


def train_clf(args, Z):
    """ calc classifier loss """
    (clf_acc, clf_loss), _ = ops.clf_loss(args, Z)
    clf_loss = clf_loss * args.beta
    return clf_acc, clf_loss


def train_gen(args, netG):
    netG.zero_grad()
    z = sample_z(args)
    fake = netG(z)
    G = netD(fake).mean()
    G_cost = -G
    return G


def train(args):
    
    netE = Encoder_fc(args).cuda()
    netG = Generator_fc(args).cuda()
    netD = Discriminator_fc(args).cuda()
    print (netE, netG, netD)

    optimizerE = optim.Adam(netE.parameters(), lr=1e-3, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-3, betas=(0.5, 0.9))
    optimizerD = optim.Adam(netD.parameters(), lr=1e-3, betas=(0.5, 0.9))
    
    if args.use_wae:
        netDz = Discriminator_wae(args, shape).cuda()
        optimizerDz = optim.Adam(netDz.parameters(), lr=1e-3, betas=(0.5, 0.9))
    
    base_gen = []
    param_gen = []
    base_gen.append((utils.dataset_iterator(args, 0)))
    base_gen.append((utils.dataset_iterator(args, 1)))
    param_gen.append(utils.inf_train_gen(base_gen[0][0]))
    param_gen.append(utils.inf_train_gen(base_gen[1][0]))
    
    print ('==> created data generators')
    torch.manual_seed(1)

    for iteration in range(0, args.epochs):
        start_time = time.time()

        """ Update AE """
        print ("==> autoencoding layers")
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        for id in range(args.stat['n_layers']):
            print ("- layer ", id)
            args.id = id
            x = sample_x(args, param_gen[id], id)
            ae_loss = train_ae(args, netG, netE, x)
            ae_loss.backward()
        optimizerE.step()
        optimizerG.step()
        print ('==> updated AE') 

        """ Update Adversary """
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
        
        #for iter_d in range(5):
        x = sample_x(args, param_gen[id], id)
        z = ops.gen_layer(args, netG, sample_z(args))
        for id in range(args.stat['n_layers']):
            args.id = id
            print ('layer : ', id)
            d_real, d_fake, gp = train_adv(args, netD, x, z)
            d_real.backward(torch.Tensor([-1]).cuda())
            d_fake.backward()
            gp.backward()
            w1 = d_real - d_fake
            layers.append(z)
            z = ops.gen_layer(netE(z))  # Here we go

        acc, loss = train_clf(args, layers)
        clf_loss.backward()
        d_cost = d_fake - d_real + gp + clf_loss
        optimizerD.step()
            
        print ('==> updated D')
        if args.use_wae:
            train_wadv(args, netDz, netE, x, z)
            optimizerDz.step()
            print ('==> updated Dz')

        for id in range(args.stat['n_layers']):
            args.id = id
            g_cost = train_gen(args, netG)
            g_cost.backward(torch.Tensor([-1]).cuda())
        optimizerG.step()

        print ('==> updated G')
        if iteration % 10 == 0:
            print ('==> iter: ', iteration)
        # Write logs
        if iteration % 100 == 0:
            save_dir = './plots/{}/{}'.format(args.dataset, args.model)
            path = 'params/sampled/{}/{}'.format(args.dataset, args.model)
            utils.save_model(args, netE, optimizerE)
            utils.save_model(args, netG, optimizerG)
            utils.save_model(args, netD, optimizerD)
            print ("==> saved model instances")
            if not os.path.exists(path):
                os.makedirs(path)
            # samples = netG(z)
            samples = []
            l = ops.gen_layer(args, netG, z)
            for id in args.stat['n_layers']:
                args.id = id
                z = sample_z(args)
                samples.append(l)
                l = ops.gen_layer(netE(l))
            (acc, loss), (oracle_acc, oracle_loss) = ops.clf_loss(args, samples)
            print ("****************")
            print('Iter ', iteration, 'Beta ', args.beta)
            print('D cost', d_cost.cpu().data.numpy()[0])
            print('G cost', g_cost.cpu().data.numpy()[0])
            print('AE cost', ae_loss.cpu().data.numpy()[0])
            print('W1 distance', w1.cpu().data.numpy()[0])
            print ('clf -> oracle (acc)', clf_acc, oracle_acc)
            print ('clf -> oracle (loss)', clf_loss/args.beta, oracle_loss/args.beta)
            print ('filter 1: ', samples[0, 0, :, :])
            print ("****************")


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

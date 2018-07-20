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
    parser.add_argument('-s', '--model', default='fcn2', type=str)
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

        self.linear1 = nn.Linear(self.lcd, nfe*4)
        self.linear2 = nn.Linear(self.nfe*4, self.nfe*2)
        self.linear3 = nn.Linear(self.nfe*2, self.dim)
        self.relu = nn.LeakyReLU(.2, inplace=True)

    def forward(self, x):
        print ('E in: ', x.shape)
        x = x.view(-1, self.ng)
        #if self.is_training:
        #    z = torch.normal(torch.zeros_like(x.data), std=0.01)
        #    x.data += z
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

        self.linear1 = nn.Linear(self.dim, self.nfg*4)
        self.linear2 = nn.Linear(self.nfg*4, self.nfg*2)
        self.linear3 = nn.Linear(self.nfg*2, self.nfg)
        self.linear_out = nn.Linear(self.nfg, self.lcd)
        self.relu = nn.LeakyReLU(.2, inplace=True)

    def forward(self, x):
        print ('G in: ', x.shape)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear_out(x)
        x = x.view(-1, self.nf, self.nf)
        print ('G out: ', x.shape)
        return x


class Generator_conv(nn.Module):
    def __init__(self, args, datashape):
        super(Generator_conv, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)

        self.linear = nn.Linear(nfg, 64*64)
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
    def __init__(self, args,):
        super(Discriminator_fc, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)

        self.ng = self.batch_size * self.lcd
        self.linear1 = nn.Linear(self.ng, self.nfd)
        self.linear2 = nn.Linear(self.nfd, self.nfd)
        self.linear3 = nn.Linear(self.nfd, 1)
        self.relu = nn.LeakyReLU(.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print ('D in: ', x.shape)
        x = x.view(-1, self.ng)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.sigmoid(x)
        print ('D out: ', x.shape)
        return x

class Discriminator_z_fc(nn.Module):
    def __init__(self, args, datashape):
        super(Discriminator_z_fc, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)

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


def calc_gradient_penalty(args, shape, netD, real_data, gen_data):
    batch_size = args.batch_size
    datashape = (128, 256, 7, 7)

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


def load_networks(args):
args = load_args()
modeldef = netdef.nets()[args.model]
args.lcd = modeldef['base_shape']

print (modeldef)

netE = Encoder_fc(args, modeldef)
netG = Generator_fc(args, modeldef)
netD = Discriminator_fc(args, modeldef)


train_gen, dev_gen = utils.dataset_iterator(args)
torch.manual_seed(1)

optimizerE = optim.Adam(netE.parameters(), lr=1e-3, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-3, betas=(0.5, 0.9))
optimizerD = optim.Adam(netD.parameters(), lr=1e-3, betas=(0.5, 0.9))


if use_wae:
    netDz = Discriminator_wae(args, shape).cuda()
    optimizerDz = optim.Adam(netDz.parameters(), lr=1e-3, betas=(0.5, 0.9))

ae_criterion = nn.MSELoss()

one = torch.FloatTensor([1]).cuda()
mone = (one * -1).cuda()
gen = utils.inf_train_gen(train_gen)

for iteration in range(0, 100000):
    start_time = time.time()

    """ Update AE """
    for p in netD.parameters():
        p.requires_grad = False  # to avoid computation
    netG.zero_grad()
    netE.zero_grad()

    _data = next(gen)
    real_data = torch.Tensor(_data).cuda()
    real_data_v = autograd.Variable(real_data)
    encoding  = netE(real_data_v)
    # generate fake layer
    fake = ops.gen_layer(args, netG, encoding)
    # fake = netG(encoding)
    ae_loss = ae_criterion(fake, real_data_v)
    ae_loss.backward(one)
    optimizerE.step()
    optimizerG.step()
     
    """ Update Adversary """
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update
    for iter_d in range(5):
        real_data = torch.Tensor(_data).cuda()
        real_data_v = autograd.Variable(real_data)
        netD.zero_grad()

        """ Update Dz """
        if use_wae:
            netDz.zero_grad()
            z_real = autograd.Variable(torch.randn(args.batch_size, args.dim)).cuda()
            z_fake = netE(real_data_v).view(args.batch_size, -1)
            Dz_real = netDz(z_real)
            Dz_fake = netDz(z_fake)
            Dz_loss = -(torch.mean(Dz_real) - torch.mean(Dz_fake))
            Dz_loss.backward()
            optimizerDz.step()

        """ update Dg """
        D_real = netD(real_data_v)
        D_real = D_real.mean()
        D_real.backward(mone)
        noise = torch.randn(args.batch_size, args.dim).cuda()
        noisev = autograd.Variable(noise)
        fake = ops.gen_layer(args, netG, noisev)
        # fake = netG(noisev)
        D_fake = netD(fake)
        D_fake = D_fake.mean()
        D_fake.backward(one)
        # train with gradient penalty

        gradient_penalty = ops.calc_gradient_penalty(args, netD,
                real_data_v.data, fake.data)
        gradient_penalty.backward()

        """ calc classifier loss """
        clf_acc, clf_loss = ops.clf_loss(args, iteration, fake)
        D_cost = D_fake - D_real + gradient_penalty + clf_loss
        Wasserstein_D = D_real - D_fake
        optimizerD.step()

    """ Update Generator network """
    noise = torch.randn(args.batch_size, args.dim).cuda()
    noisev = autograd.Variable(noise)
    fake = netG(noisev)
    G = netD(fake)
    G = G.mean()
    G.backward(mone)
    G_cost = -G
    optimizerG.step()

    # Write logs and save samples
    save_dir = './plots/{}/{}/{}'.format(args.dataset, args.size, args.layer)
    # Calculate dev loss and generate samples every 100 iters
    if iteration % 10 == 0:
        print ('==> iter: ', iteration)
    if iteration % 100 == 0:
        utils.save_model(netG, optimizerG, iteration,
                'WGAN/{}/Generators/{}G_{}'.format(
                    args.dataset, args.size, iteration))
        utils.save_model(netD, optimizerD, iteration, 
                'WGAN/{}/Discriminators/{}D_{}'.format(
                    args.dataset, args.size, iteration))
        print ("==> saved model instances")
        dev_disc_costs = []
        for params in dev_gen():
            p = torch.Tensor(params).cuda()
            p_v = autograd.Variable(p, volatile=True)
            D = netD(p_v)
            _dev_disc_cost = -D.mean().cpu().data.numpy()
            dev_disc_costs.append(_dev_disc_cost)
        
        path = 'params/sampled/{}/{}/{}'.format(args.dataset, args.size, args.layer)
        if not os.path.exists(path):
            os.makedirs(path)
        z = torch.randn(args.batch_size, args.dim).cuda()
        z = autograd.Variable(z)
        samples = ops.gen_layer(args, netG, z)
        # samples = netG(z)
        acc, loss = ops.clf_loss(args, iteration, samples)
        utils.save_samples(args, samples, iteration, path)
        # acc = utils.generate_samples(iteration, netG, path, args)
        if args.comet:        
            experiment.log_metric('train D cost', D_cost.cpu().data.numpy()[0])
            experiment.log_metric('train G cost', G_cost.cpu().data.numpy()[0])
            experiment.log_metric('AE cost', ae_loss.cpu().data.numpy()[0])
            experiment.log_metric('W1 distance', Wasserstein_D.cpu().data.numpy()[0])
            experiment.log_metric('dev D cost', np.mean(dev_disc_costs))
            experiment.log_metric('{} accuracy'.format(args.dataset), acc)
            experiment.log_metric('{} loss'.format(args.dataset), loss)
        
        print ("****************")
        print('Iter ', iteration, 'Beta ', args.beta)
        print('D cost', D_cost.cpu().data.numpy()[0])
        print('G cost', G_cost.cpu().data.numpy()[0])
        print('AE cost', ae_loss.cpu().data.numpy()[0])
        print('W1 distance', Wasserstein_D.cpu().data.numpy()[0])
        print ('clf accuracy', clf_acc)
        print ('clf loss', clf_loss/args.beta)
        # print sample filter
        print ('filter 1: ', samples[0, 0, :, :])
        print ("****************")


if __name__ == '__main__':
    train()

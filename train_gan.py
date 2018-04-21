import sys
import time
import argparse
import numpy as np
from glob import glob
from scipy.misc import imshow
from comet_ml import Experiment
import comet_params    
import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
from torch.nn import functional as F

import plot
import utils
import generators
import discriminators


def load_args():

    parser = argparse.ArgumentParser(description='param-wgan')
    parser.add_argument('-z', '--dim', default=64, type=int, help='latent space size')
    parser.add_argument('-g', '--gp', default=10, type=int, help='gradient penalty')
    parser.add_argument('-b', '--batch_size', default=64, type=int)
    parser.add_argument('-e', '--epochs', default=200000, type=int)
    parser.add_argument('-o', '--output_dim', default=784, type=int)
    parser.add_argument('-m', '--model', default='conv', type=str)
    parser.add_argument('-s', '--size', default='wide', type=str)
    parser.add_argument('-d', '--dataset', default='cifar', type=str)
    parser.add_argument('-l', '--layer', default='conv2', type=str)

    args = parser.parse_args()
    return args


def load_models(args):
    shapes = {'mwideconv1': (128, 3, 3), 'mwideconv2': (256, 3, 3),
              'mwide7conv1': (128, 7, 7), 'mwide7conv2': (256, 7, 7),
              'cwideconv1': (640, 3, 3), 'cwideconv2': (1280, 3, 3),
              'm1xconv1': (32, 3, 3), 'm1xconv2': (64, 3, 3),
              'c1xconv1': (64, 3, 3), 'c1xconv2': (128, 3, 3)}

    if args.dataset == 'mnist':
        if args.size == '1x':
            if args.layer == 'conv1':
                netG = generators.MGenerator(args, shapes['m1xconv1']).cuda()
                netD = discriminators.Discriminator(args, shapes['m1xconv1']).cuda()
            if args.layer == 'conv2':
                netG = generators.MGenerator(args, shapes['m1xconv2']).cuda()
                netD = discriminators.Discriminator(args, shapes['m1xconv2']).cuda()
        elif args.size == 'wide':
            if args.layer == 'conv1':
                netG = generators.MGeneratorWide(args, shapes['mwideconv1']).cuda()
                netD = discriminators.DiscriminatorWide(args, shapes['mwideconv1']).cuda()
            if args.layer == 'conv2':
                netG = generators.MGeneratorWide(args, shapes['mwideconv2']).cuda()
                netD = discriminators.DiscriminatorWide(args, shapes['mwideconv2']).cuda()
        elif args.size == 'wide7':
            if args.layer == 'conv1':
                netG = generators.MGeneratorWide7(args, shapes['mwide7conv1']).cuda()
                netD = discriminators.DiscriminatorWide7(args, shapes['mwide7conv1']).cuda()
            if args.layer == 'conv2':
                netG = generators.MGeneratorWide7(args, shapes['mwide7conv2']).cuda()
                netD = discriminators.DiscriminatorWide7(args, shapes['mwide7conv2']).cuda()


    elif args.dataset =='cifar':
        netG = generators.CGenerator(args).cuda()
        netD = discriminators.Discriminator(args).cuda()
    print (netD, netG)
    return (netD, netG)


def train():
    args = load_args()
    experiment = Experiment(api_key="54kuR3NpJIb6ibDx9HVSbbHdw", project_name="paramgan")
    experiment.log_multiple_params(comet_params.get_hyper_params())

    train_gen, dev_gen = utils.dataset_iterator(args)
    torch.manual_seed(1)
    netD, netG = load_models(args)
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
    one = torch.FloatTensor([1]).cuda()
    mone = (one * -1).cuda()
    gen = utils.inf_train_gen(train_gen)

    for iteration in range(100000):
        start_time = time.time()
        if iteration % 1000 == 0:
            # if we are trining more nets, scan again for more data
            train_gen, dev_gen = utils.dataset_iterator(args)
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
            # Update Discriminator
        for iter_d in range(5):
            _data = next(gen)
            real_data = torch.Tensor(_data).cuda()
            real_data_v = autograd.Variable(real_data)
            netD.zero_grad()
            # train with real
            D_real = netD(real_data_v)
            D_real = D_real.mean()
            # print D_real
            D_real.backward(mone)
            # train with fake
            noise = torch.randn(args.batch_size, args.dim).cuda()
            noisev = autograd.Variable(noise, volatile=True)  # totally freeze netG
            fake = autograd.Variable(netG(noisev).data)
            inputv = fake
            D_fake = netD(inputv)
            D_fake = D_fake.mean()
            D_fake.backward(one)
            # train with gradient penalty
            gradient_penalty = utils.calc_gradient_penalty(args, 
                    netD, real_data_v.data, fake.data)
            gradient_penalty.backward()

            D_cost = D_fake - D_real + gradient_penalty
            Wasserstein_D = D_real - D_fake
            optimizerD.step()

        # Update Generator network
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        netG.zero_grad()

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
        if iteration % 100 == 0:
            dev_disc_costs = []
            for params in dev_gen():
                p = torch.Tensor(params).cuda()
                p_v = autograd.Variable(p, volatile=True)
                D = netD(p_v)
                _dev_disc_cost = -D.mean().cpu().data.numpy()
                dev_disc_costs.append(_dev_disc_cost)
            
            acc = utils.generate_samples(iteration,
                                         netG,
                                         'params/sampled/{}/{}/{}'.format(
                                             args.dataset, args.size, args.layer),
                                         args)

            experiment.log_metric('train D cost', D_cost.cpu().data.numpy()[0])
            experiment.log_metric('train G cost', G_cost.cpu().data.numpy()[0])
            experiment.log_metric('W1 distance', Wasserstein_D.cpu().data.numpy()[0])
            experiment.log_metric('dev D cost', np.mean(dev_disc_costs))
            experiment.log_metric('{} accuracy'.format(args.dataset), acc)
            print ("****************")
            print('D cost', D_cost.cpu().data.numpy()[0])
            print('G cost', G_cost.cpu().data.numpy()[0])
            print('W1 distance', Wasserstein_D.cpu().data.numpy()[0])
            print ('{} accuracy'.format(args.dataset), acc)
            print ("****************")


if __name__ == '__main__':
    train()

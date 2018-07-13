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
import encoders
import generators
import discriminators


def load_args():

    parser = argparse.ArgumentParser(description='param-wgan')
    parser.add_argument('-z', '--dim', default=128, type=int, help='latent space size')
    parser.add_argument('-g', '--gp', default=10, type=int, help='gradient penalty')
    parser.add_argument('-b', '--batch_size', default=256, type=int)
    parser.add_argument('-e', '--epochs', default=200000, type=int)
    parser.add_argument('-o', '--output_dim', default=784, type=int)
    parser.add_argument('-m', '--model', default='conv', type=str)
    parser.add_argument('-s', '--size', default='wide7', type=str)
    parser.add_argument('-d', '--dataset', default='mnist', type=str)
    parser.add_argument('-l', '--layer', default='conv2', type=str)
    parser.add_argument('--nf', default=128, type=int)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--beta', default=10e3, type=float)
    parser.add_argument('--comet', default=False, type=bool)

    args = parser.parse_args()
    return args


def load_models(args):
    shapes = {'mwideconv1': (128, 3, 3), 'mwideconv2': (256, 3, 3),
              'mwide7conv1': (128, 7, 7), 'mwide7conv2': (256, 7, 7),
              'cwideconv1': (640, 3, 3), 'cwideconv2': (1280, 3, 3),
              'm1xconv1': (32, 3, 3), 'm1xconv2': (64, 3, 3),
              'c1xconv1': (64, 3, 3), 'c1xconv2': (128, 3, 3),
              'resnet': (512, 3, 3)}

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
                netE = encoders.EncoderWide7FC(args, shapes['mwide7conv1']).cuda()
                netG = generators.GeneratorWide7FC(args, shapes['mwide7conv1']).cuda()
                netD = discriminators.DiscriminatorWide7FC(args, shapes['mwide7conv1']).cuda()
            if args.layer == 'conv2':
                netE = encoders.EncoderWide7FC(args, shapes['mwide7conv2']).cuda()
                netG = generators.GLayer7FC(args, shapes['mwide7conv2']).cuda()
                netD = discriminators.DiscriminatorWide7FC(args, shapes['mwide7conv2']).cuda()


    elif args.dataset =='cifar':
        if args.size in ['presnet', 'resnet']:
            netG = generators.ResNetGenerator(args).cuda()
            print (netG)
            netD = discriminators.ResNetDiscriminator(args, shapes['resnet']).cuda()
            print (netD)
        if args.size == '1x':
            netG = generators.CGenerator(args).cuda()
            netD = discriminators.Discriminator(args).cuda()
    print (netE, netD, netG)
    return (netE, netD, netG)


def train():
    args = load_args()
    if args.comet:
        experiment = Experiment(api_key="54kuR3NpJIb6ibDx9HVSbbHdw", project_name="HyperGAN")
    train_gen, dev_gen = utils.dataset_iterator(args)
    torch.manual_seed(1)
    
    netE, netD, netG = load_models(args)
    optimizerE = optim.Adam(netE.parameters(), lr=1e-3, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-3, betas=(0.5, 0.9))
    optimizerD = optim.Adam(netD.parameters(), lr=1e-3, betas=(0.5, 0.9))
    ae_criterion = nn.MSELoss()

    if args.resume is True:
        print ('==> reusing old weights if possible')
        try:
            netE, optimE, _ = utils.load_model(netE, optimE, "E_latest.pth")
            netG, optimG, _ = utils.load_model(netG, optimG, "G_latest.pth")
            netD, optimD, _ = utils.load_model(netD, optimD, "D_latest.pth")
            print ("Loaded all models successfully, Proceeding...")
        except:
            print ("Model Loading Failed, Proceeding...")

    one = torch.FloatTensor([1]).cuda()
    mone = (one * -1).cuda()
    gen = utils.inf_train_gen(train_gen)

    for iteration in range(0, 100000):
        start_time = time.time()
        #if iteration % 1000 == 0:
        #    # if we are trining more nets, scan again for more data
        #    train_gen, dev_gen = utils.dataset_iterator(args)

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
        ae_loss = ae_criterion(fake, real_data_v)
        ae_loss.backward(one)
        optimizerE.step()
        optimizerG.step()
        
        """ Update Adversary """
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
            # Update Discriminator
        for iter_d in range(1):
            real_data = torch.Tensor(_data).cuda()
            real_data_v = autograd.Variable(real_data)
            netD.zero_grad()
            # train with real
            D_real = netD(real_data_v)
            D_real = D_real.mean()
            D_real.backward(mone)
            # train with fake
            noise = torch.randn(args.batch_size, args.dim).cuda()
            noisev = autograd.Variable(noise)
            fake = ops.gen_layer(args, netG, noisev)
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

        # Update Generator network
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
            print('Iter ', iteration)
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

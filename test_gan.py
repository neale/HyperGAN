import os
import numpy as np
import torch
import argparse
import torch.backends.cudnn as cudnn
import utils
import generators
import discriminators
from torch import autograd

def load_args():

    parser = argparse.ArgumentParser(description='param-wgan')
    parser.add_argument('--dpath', help='discriminator weights')
    parser.add_argument('--gpath', help='generator weights')
    parser.add_argument('-i', '--iters', default=10, type=int, help='n samples')
    parser.add_argument('-b', '--batch_size', default=64, type=int)
    parser.add_argument('-m', '--model', default='conv', type=str)
    parser.add_argument('-s', '--size', default='wide', type=str)
    parser.add_argument('-d', '--dataset', default='cifar', type=str)
    parser.add_argument('-l', '--layer', default='conv2', type=str)
    parser.add_argument('--dim', default=64, type=int)

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
                netG = generators.MGeneratorWide7(args, shapes['mwide7conv1']).cuda()
                netD = discriminators.DiscriminatorWide7(args, shapes['mwide7conv1']).cuda()
            if args.layer == 'conv2':
                netG = generators.MGeneratorWide7(args, shapes['mwide7conv2']).cuda()
                netD = discriminators.DiscriminatorWide7(args, shapes['mwide7conv2']).cuda()


    elif args.dataset =='cifar':
        if args.size == 'presnet':
            netG = generators.ResNetGenerator(args).cuda()
            netD = discriminators.ResNetDiscriminator(args, shapes['resnet']).cuda()
        if args.size == '1x':
            netG = generators.CGenerator(args).cuda()
            netD = discriminators.Discriminator(args).cuda()
    print (netD, netG)
    return (netD, netG)


def test():
    args = load_args()
    train_gen, dev_gen = utils.dataset_iterator(args)
    torch.manual_seed(1)
    netD, netG = load_models(args)
    one = torch.FloatTensor([1]).cuda()
    mone = (one * -1).cuda()
    gen = utils.inf_train_gen(train_gen)

    print ("Loading checkpoints...")
    checkpointG = torch.load(args.gpath)
    checkpointD = torch.load(args.dpath)

    netG.load_state_dict(checkpointG['state_dict'])
    netD.load_state_dict(checkpointD['state_dict'])

    cudnn.benchmark = True

    for iteration in range(args.iters):

        dev_disc_costs = []
        for params in dev_gen():
            p = torch.Tensor(params).cuda()
            p_v = autograd.Variable(p, volatile=True)
            D = netD(p_v)
            _dev_disc_cost = -D.mean().cpu().data.numpy()
            dev_disc_costs.append(_dev_disc_cost)

        acc = utils.generate_samples(iteration, netG, 'params/test/', args)
        print ("****************")
        print('Iter ', iteration)
        print('dev D cost', np.mean(dev_disc_costs))
        print ('{} accuracy'.format(args.dataset), acc)
        print ("****************")


if __name__ == '__main__':
    test()

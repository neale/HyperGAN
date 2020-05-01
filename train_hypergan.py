""" 
this file runs the main hypergan training
the weight generators predict both weights and biases for the target network
by defaut, weight posteriors are plotted, though this can be safely commented out
"""
import matplotlib
matplotlib.use('agg')
import sys
import torch
import pprint
import argparse
import numpy as np
import importlib

import ops
import utils
import datagen
import experiments

import torch
import torch.optim
import torch.nn.functional as F

def load_args():

    parser = argparse.ArgumentParser(description='HyperGAN')
    parser.add_argument('--z', default=64, type=int, help='Q(z|s) latent space width')
    parser.add_argument('--s', default=256, type=int, help='S sample dimension')
    parser.add_argument('--bias', action='store_true', help='Include HyperGAN bias')
    parser.add_argument('--batch_size', default=100, type=int, help='network batch size')
    parser.add_argument('--epochs', default=200000, type=int)
    parser.add_argument('--target', default='small', type=str, help='target name')
    parser.add_argument('--beta', default=1, type=int, help='lagrangian strength')
    parser.add_argument('--pretrain_e', action='store_true')
    parser.add_argument('--exp', default='0', type=str)
    parser.add_argument('--resume', default=None, type=str, help='resume from path')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--wd', default=5e-4, type=float, help='weight decay (optimizer)')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--dataset', default='mnist', type=str, help='mnist, cifar, cifar_hidden')
    args = parser.parse_args()
    return args


def set_ngen(args):
    if args.target == 'small':
        args.ngen = 3
    elif args.target in ['lenet', 'mednet']: 
        args.ngen = 5
    else:
        raise ValueError
    return


def train(args):
    torch.manual_seed(1)
    set_ngen(args)
    models = importlib.import_module('models.{}'.format(args.target))
    if args.cuda and torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    """ instantiate HyperGAN """
    hypergan = models.HyperGAN(args)
    generator = hypergan.generator
    mixer = hypergan.mixer
    Dz = hypergan.discriminator
    if args.resume is not None:
        hypergan.restore_models()
    print (mixer, generator.as_list(), Dz)
    
    """ attach optimizers """
    optimQ = torch.optim.Adam(mixer.parameters(), lr=args.lr, weight_decay=args.wd)
    optimW = []
    for m in range(args.ngen):
        s = getattr(generator, 'W{}'.format(m+1))
        optimW.append(torch.optim.Adam(s.parameters(), lr=args.lr, weight_decay=args.wd))
    optimD = torch.optim.Adam(Dz.parameters(), lr=args.lr, weight_decay=args.wd)
    schedulers = []
    steps = [10*i for i in range(1, 100)]
    for op in [optimQ, optimD, *optimW]:
        schedulers.append(utils.CyclicCosAnnealingLR(op, steps, eta_min=1e-8))
    best_test_acc, best_test_loss, = 0., np.inf
    args.best_loss, args.best_acc = best_test_loss, best_test_acc

    trainset, testset = getattr(datagen, 'load_{}'.format(args.dataset))()

    if args.pretrain_e is True:
       print ("==> pretraining encoder")
       ops.pretrain_encoder(args, mixer, optimQ)
       
    getattr(experiments, 'sample_weight_posteriors_{}'.format(args.target))(args, hypergan, 0)
    print ('==> Begin Training')
    for epoch in range(args.epochs):
        for batch_idx, (data, target) in enumerate(trainset):
            """ Update discriminator on each sample from Q(z|s) """
            s = torch.randn(args.batch_size, args.s).to(args.device)
            z = torch.randn(args.batch_size, args.z).to(args.device)
            full_z = torch.randn(args.z*args.ngen, args.batch_size)
            codes = mixer(s)
            d_loss, d_q = ops.calc_d_loss(args, Dz, z, codes)
            d_loss = d_loss * args.beta
            optimD.zero_grad()
            d_loss.backward(retain_graph=True)
            optimD.step()

            """ generate weights ~ G(Q(s)) """
            params = generator(codes)
            """ eval weights on target architecture (training set) """
            clf_loss = 0.
            data = data.to(args.device)
            target = target.to(args.device)
            for (layers) in zip(*params):
                out = hypergan.eval_f(args, layers, data)
                loss = F.cross_entropy(out, target)
                pred = out.data.max(1, keepdim=True)[1]
                acc = pred.eq(target.data.view_as(pred)).long().cpu().sum()
                clf_loss += loss

            """ calculate total loss on Q and G """
            one_qz = torch.ones((args.batch_size*args.ngen, 1), requires_grad=True).to(args.device)
            log_qz = ops.log_density(torch.ones(args.batch_size*args.ngen, 1), 2).view(-1, 1).to(args.device)
            Q_loss = F.binary_cross_entropy_with_logits(d_q+log_qz, one_qz)
            G_loss = clf_loss / args.batch_size
            QG_loss = Q_loss + G_loss
            QG_loss.backward()
            
            optimQ.step()
            for optim in optimW:
                optim.step()
            optimQ.zero_grad()
            for optim in optimW:
                optim.zero_grad()
        
        for scheduler in schedulers:
            scheduler.step()
       
        """ print training accuracy """
        print ('**************************************')
        print ('Epoch: {}'.format(epoch))
        print ('Train Acc: {}, G Loss: {}, D loss: {}'.format(acc, QG_loss, d_loss))
        print ('best test loss: {}'.format(args.best_loss))
        print ('best test acc: {}'.format(args.best_acc))
        print ('**************************************')
                        
        """ test random draw on testing set """
        test_acc = 0.
        test_loss = 0.
        with torch.no_grad():
            for i, (data, target) in enumerate(testset):
                data = data.to(args.device)
                target = target.to(args.device)
                s = torch.randn(args.s, args.batch_size).to(args.device)
                codes = mixer(s)
                params = generator(codes)
                
                for (layers) in zip(*params):
                    out = hypergan.eval_f(args, layers, data)
                    loss = F.cross_entropy(out, target)
                    pred = out.data.max(1, keepdim=True)[1]
                    correct = pred.eq(target.data.view_as(pred)).long().cpu().sum()
                    test_acc += correct.item()
                    test_loss += loss.item()

            test_loss /= len(testset.dataset) * args.batch_size
            test_acc /= len(testset.dataset) * args.batch_size
            print ('Test Accuracy: {}, Test Loss: {}'.format(test_acc, test_loss))
            if test_loss < best_test_loss:
                best_test_loss, args.best_loss = test_loss, test_loss
            if test_acc > best_test_acc:
                hypergan.save_models(args, test_acc)
            if test_acc > best_test_acc:
                best_test_acc, args.best_acc = test_acc, test_acc
        """ plot weight posteriors """
        getattr(experiments, 'sample_weight_posteriors_{}'.format(args.target))(args, hypergan, epoch)
        
       
if __name__ == '__main__':
    args = load_args()
    train(args)

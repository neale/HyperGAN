import os
import sys
import time
import argparse
import numpy as np
import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
from torch.nn import functional as F
import torch.distributions.multivariate_normal as N
import pprint

# import models.models_cifar as models
from models.mednet2 import HyperGAN
import models.mednet2 as models

import ops
import utils
import netdef
import datagen


def load_args():

    parser = argparse.ArgumentParser(description='param-wgan')
    parser.add_argument('--z', default=256, type=int, help='latent space width')
    parser.add_argument('--s', default=512, type=int, help='encoder dimension')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--target', default='mednet', type=str)
    parser.add_argument('--dataset', default='cifar', type=str)
    parser.add_argument('--beta', default=1., type=float)
    parser.add_argument('--exp', default='0', type=str)
    parser.add_argument('--model', default='full', type=str)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--pretrain_e', default=True, type=bool)
    parser.add_argument('--scratch', default=False, type=bool)
    parser.add_argument('--use_bn', default=True, type=bool)
    parser.add_argument('--bias', default=True, type=bool)


    args = parser.parse_args()
    return args

class HyperGANTrainer(object):
    def __init__(self, args):
        self.s = args.s
        self.z = args.z
        self.batch_size = args.batch_size
        self.epochs = 200
        self.beta = args.beta
        self.target = args.target
        self.use_bn = args.use_bn
        self.bias = args.bias
        self.pretrain_e = args.pretrain_e
        self.device = torch.device('cuda')
        torch.manual_seed(8734)        

        self.hypergan = HyperGAN(args, self.device)
        self.hypergan.print_hypergan()
        self.hypergan.attach_optimizers(5e-3, 1e-4, 5e-5)

    def train_clf(self, params, data, target, val=False):
        """ calc classifier loss """
        data, target = data.cuda(), target.cuda()
        output = self.hypergan.eval_f(params, data)
        loss = F.cross_entropy(output, target)
        correct = None
        if val:
            pred = output.data.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).float().cpu().sum()
        return (correct, loss)

    def pretrain_loss(self, code, z):
        mean_z = torch.mean(z, dim=0, keepdim=True)
        mean_e = torch.mean(code, dim=0, keepdim=True)
        mean_loss = F.mse_loss(mean_z, mean_e)
        cov_z = torch.matmul((z-mean_z).transpose(0, 1), z-mean_z)
        cov_z /= 999
        cov_e = torch.matmul((code-mean_e).transpose(0, 1), code-mean_e)
        cov_e /= 999
        cov_loss = F.mse_loss(cov_z, cov_e)
        return mean_loss, cov_loss
    
    def pretrain_encoder(self):
        j = 0
        final = 100.
        e_batch_size = 1000
        for j in range(1000):
            x = torch.randn(e_batch_size, self.s).cuda()
            z = torch.randn(e_batch_size, self.z).cuda()
            codes = self.hypergan.mixer(x)
            for i, code in enumerate(codes):
                code = code.view(e_batch_size, self.z)
                mean_loss, cov_loss = self.pretrain_loss(code, z)
                loss = mean_loss + cov_loss
                loss.backward(retain_graph=True)
            self.hypergan.optim_mixer.step()
            self.hypergan.mixer.zero_grad()

            print ('Pretrain Enc iter: {}, Mean Loss: {}, Cov Loss: {}'.format(
                j, mean_loss.item(), cov_loss.item()))
            final = loss.item()
            if loss.item() < 0.1:
                print ('Finished Pretraining Encoder')
                break
        
    def train(self):
        cifar_train, cifar_test = datagen.load_cifar()
        best_test_acc, best_test_loss = 0., np.inf
        self.best_loss, self.best_acc = best_test_loss, best_test_acc

        one = torch.FloatTensor([1]).cuda()
        mone = (one * -1).cuda()
        if self.pretrain_e:
            print ("==> pretraining encoder")
            self.pretrain_encoder()

        print ('==> Begin Training')
        for epoch in range(1000):
            for batch_idx, (data, target) in enumerate(cifar_train):
                s = torch.randn(self.batch_size, self.s).cuda()
                codes = self.hypergan.mixer(s)
                params = self.hypergan.generator(codes)
                
                # Z Adversary
                
                for code in codes:
                    noise = torch.randn(self.batch_size, self.z).cuda()
                    d_real = self.hypergan.discriminator(noise)
                    d_fake = self.hypergan.discriminator(code)
                    d_real_loss = -1 * torch.log((1-d_real).mean())
                    d_fake_loss = -1 * torch.log(d_fake.mean())
                    d_real_loss.backward(retain_graph=True)
                    d_fake_loss.backward(retain_graph=True)
                    d_loss = d_real_loss + d_fake_loss
                self.hypergan.optim_disc.step()
                
                d_loss = 0
                losses, corrects = [], []
                for (layers) in zip(*params):
                    correct, loss = self.train_clf(layers, data, target, val=True)
                    losses.append(loss)
                    corrects.append(correct)
                loss = torch.stack(losses).mean()
                correct = torch.stack(corrects).mean()
                scaled_loss = self.beta * loss
                scaled_loss.backward()
                   
                self.hypergan.optim_mixer.step()
                self.hypergan.update_generator()
                self.hypergan.zero_grad()
                
                loss = loss.item()
                
                """ Update Statistics """
                if batch_idx % 100 == 0:
                    acc = (100* (correct / self.batch_size))
                    print ('**************************************')
                    print ('CIFAR Test, epoch: {}'.format(epoch))
                    print ('Acc: {}, G Loss: {}, D Loss: {}'.format(acc, loss, d_loss))
                    print ('best test loss: {}'.format(self.best_loss))
                    print ('best test acc: {}'.format(self.best_acc))
                    print ('**************************************')
                
            with torch.no_grad():
                test_acc = 0.
                test_loss = 0.
                total_correct = 0.
                for i, (data, target) in enumerate(cifar_test):
                    z = torch.randn(self.batch_size, self.s).cuda()
                    codes = self.hypergan.mixer(z)
                    params = self.hypergan.generator(codes)
                    
                    losses, corrects = [], []
                    for (layers) in zip(*params):
                        correct, loss = self.train_clf(layers, data, target, val=True)
                        losses.append(loss)
                        corrects.append(correct)
                    losses = torch.stack(losses)
                    corrects = torch.stack(corrects)
                    test_acc += corrects.mean().item()
                    total_correct += corrects.sum().item()
                    test_loss += losses.mean().item()
                test_loss /= len(cifar_test.dataset)
                test_acc /= len(cifar_test.dataset)
                total_correct /= self.batch_size
                
                print ('[Epoch {}] Test Loss: {}, Test Accuracy: {},  ({}/{})'.format(epoch, test_loss, test_acc,
                    total_correct, len(cifar_test.dataset)))

                if test_loss < best_test_loss or test_acc > best_test_acc:
                    print ('==> new best stats, saving')
                    if test_loss < best_test_loss:
                        best_test_loss = test_loss
                        self.best_loss = test_loss
                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
                        self.best_acc = best_test_acc


if __name__ == '__main__':

    args = load_args()
    modeldef = netdef.nets()[args.target]
    pprint.pprint (modeldef)
    # log some of the netstat quantities so we don't subscript everywhere
    args.stat = modeldef
    args.shapes = modeldef['shapes']
    args.lcd = modeldef['base_shape']
    # why is a running product so hard in python
    args.gcd = int(np.prod([*args.shapes[0]]))
    trainer = HyperGANTrainer(args)
    trainer.train()


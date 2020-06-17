import os
import sys
import time
import argparse
import numpy as np
import torch
import torchvision
from torch import nn
from torch import autograd

import itertools

from torch.nn import functional as F
import torch.distributions.multivariate_normal as N
import pprint

# import models.models_cifar as models
#from models.mednet2 import HyperGAN
from models.lenet import HyperGAN
import models.lenet as models

import ops
import utils
import netdef
import datagen


def load_args():

    parser = argparse.ArgumentParser(description='param-wgan')
    parser.add_argument('--z', default=256, type=int, help='latent space width')
    parser.add_argument('--s', default=512, type=int, help='encoder dimension')
    parser.add_argument('--n_hidden', default=100, type=int, help='g hidden dimension')
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--target', default='lenet', type=str)
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--beta', default=1., type=float)
    parser.add_argument('--exp', default='0', type=str)
    parser.add_argument('--model', default='full', type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--pretrain_e', action='store_true')
    parser.add_argument('--scratch', action='store_true')
    parser.add_argument('--use_bn', action='store_true')
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--vote', default='hard', type=str)

    args = parser.parse_args()
    return args

class HyperGANTrainer(object):
    def __init__(self, args):
        self.s = args.s
        self.z = args.z
        self.batch_size = args.batch_size
        self.epochs = 200
        self.alpha = 1
        self.beta = args.beta
        self.target = args.target
        self.use_bn = args.use_bn
        self.bias = args.bias
        self.pretrain_e = args.pretrain_e
        self.dataset = args.dataset
        self.test_ensemble = args.ensemble
        self.vote = args.vote

        self.device = torch.device('cuda')
        torch.manual_seed(8734)        

        self.hypergan = HyperGAN(args, self.device)
        self.hypergan.print_hypergan()
        self.hypergan.attach_optimizers(5e-3, 1e-4, 5e-5)

        if self.dataset == 'mnist':
            self.data_train, self.data_test = datagen.load_mnist()
        elif self.dataset == 'cifar':
            self.data_train, self.data_test = datagen.load_cifar()

        self.best_test_acc = 0.
        self.best_test_loss = np.inf


    def train_clf(self, params, data, target, val=False):
        """ calc classifier loss """
        data, target = data.to(self.device), target.to(self.device)
        output = self.hypergan.eval_f(params, data)
        loss = F.cross_entropy(output, target)
        correct = None
        if val:
            pred = output.data.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).float().cpu().sum()
        return (correct, loss, output)

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
            x = torch.randn(e_batch_size, self.s).to(self.device)
            z = torch.randn(e_batch_size, self.z).to(self.device)
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
        best_test_acc, best_test_loss = 0., np.inf
        self.best_loss, self.best_acc = best_test_loss, best_test_acc

        one = torch.FloatTensor([1]).cuda()
        mone = (one * -1).cuda()
        if self.pretrain_e:
            print ("==> pretraining encoder")
            self.pretrain_encoder()

        print ('==> Begin Training')
        for epoch in range(1000):
            for batch_idx, (data, target) in enumerate(self.data_train):
                target = target.to(self.device)
                s = torch.randn(self.batch_size, self.s).to(self.device)
                codes = self.hypergan.mixer(s)
                params = self.hypergan.generator(codes)
                
                losses, corrects, outputs = [], [], []
                for (layers) in zip(*params):
                    correct, loss, output = self.train_clf(layers, data, target, val=True)
                    losses.append(loss)
                    corrects.append(correct)
                    outputs.append(output)
                
                outputs = torch.stack(outputs) # [p, batch, preds]

                output_i, output_j = torch.split(outputs, len(outputs)//2, dim=0) #[p/2, batch, preds]
                losses_i = torch.stack([F.cross_entropy(out_i, target, reduction='none') for out_i in output_i])
                losses_j = torch.stack([F.cross_entropy(out_j, target, reduction='none') for out_j in output_j])
                
                q_grad = autograd.grad(losses_j.sum(), inputs=output_j)[0]  # fix for ij

                qi_eps = output_i + torch.rand_like(output_i) * 1e-8
                qj_eps = output_j + torch.rand_like(output_j) * 1e-8

                kappa, grad_kappa = utils.batch_rbf(qj_eps, qi_eps) 

                p_ref = kappa.shape[0]
                kernel_logp = torch.einsum('ij, ikl->jkl', kappa, q_grad) / p_ref
                svgd = (kernel_logp + self.alpha * grad_kappa.mean(0)) # [n, theta]

                self.hypergan.zero_grad()
                autograd.backward(output_i, grad_tensors=svgd.detach())

                loss = losses_i.mean()
                correct = torch.stack(corrects).mean()
                   
                self.hypergan.optim_mixer.step()
                self.hypergan.update_generator()
                
                loss = loss.item()
                
                """ Update Statistics """
                if batch_idx % 100 == 0:
                    acc = (100* (correct / self.batch_size))
                    print ('**************************************')
                    print ('{} train, epoch: {}'.format(args.dataset, epoch))
                    print ('Acc: {}, G Loss: {}'.format(acc, loss))
                    print ('best test loss: {}'.format(self.best_test_loss))
                    print ('best test acc: {}'.format(self.best_test_acc))
                    print ('**************************************')

            with torch.no_grad():
                if self.test_ensemble:
                    for ens_size in [1, 5, 10, 100]:
                        test_loss, test_acc, correct = self.test(ens_size, voting=self.vote)
                        print ('[Epoch {}] Ensemble [{}]. Test Loss: {}, Test Accuracy: {},  ({}/{})'.format(
                                epoch, ens_size, test_loss, test_acc, correct, len(self.data_test.dataset)))
                else:
                    test_loss, test_acc, correct = self.test(self.batch_size, voting=self.vote)
                    print ('[Epoch {}] Test Loss: {}, Test Accuracy: {},  ({}/{})'.format(
                            epoch, test_loss, test_acc, correct, len(self.data_test.dataset)))

            if test_loss < self.best_test_loss or test_acc > self.best_test_acc:
                print ('==> new best stats, saving')
                if test_loss < self.best_test_loss:
                    self.best_test_loss = test_loss
                if test_acc > self.best_test_acc:
                    self.best_test_acc = test_acc


    def test(self, ens_size, voting):
        test_acc = 0.
        test_loss = 0.
        total_correct = 0.
        
        self.hypergan.eval_()
        for i, (data, target) in enumerate(self.data_test):
            z = torch.randn(ens_size, self.s).to(self.device)
            codes = self.hypergan.mixer(z)
            params = self.hypergan.generator(codes)
            
            losses, corrects, outputs = [], [], []
            for (layers) in zip(*params):
                _, loss, output = self.train_clf(layers, data, target, val=True)
                losses.append(loss)
                outputs.append(output)

            losses = torch.stack(losses)
            outputs = torch.stack(outputs)

            if voting == 'soft':
                probs = F.softmax(outputs, dim=-1)  # [ens, data, 10]
                preds = probs.mean(0)  # [data, 10]
                vote = preds.argmax(-1).cpu()  # [data, 1]

            elif voting == 'hard':
                probs = F.softmax(outputs, dim=-1) #[ens, data, 10]
                preds = probs.argmax(-1).cpu()  # [ens, data, 1]
                vote = preds.mode(0)[0]  # [data, 1]

            total_correct += vote.eq(target.cpu().data.view_as(vote)).float().cpu().sum()
            
            test_loss += losses.mean().item()
        test_loss /= len(self.data_test.dataset)
        test_acc = total_correct/len(self.data_test.dataset)
        self.hypergan.train_()
        
        return test_loss, test_acc, total_correct


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


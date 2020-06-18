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

#from models.mednet2 import HyperGAN
#import models.mednet2 as models
import models.lenet as models

import ops
import utils
import netdef
import datagen
import evaluate_uncertainty as uncertainty

def load_args():

    parser = argparse.ArgumentParser(description='param-wgan')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)
    parser.add_argument('--n_models', default=10, type=int)
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--exp', default='0', type=str)
    parser.add_argument('--model', default='full', type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--test_uncertainty', action='store_true')
    parser.add_argument('--vote', default='hard', type=str)

    args = parser.parse_args()
    return args

class EnsembleTrainer(object):
    def __init__(self, args):
        self.lr = args.lr
        self.wd = args.wd
        self.epochs = 200
        self.dataset = args.dataset
        self.test_uncertainty = args.test_uncertainty
        self.vote = args.vote
        self.n_models = args.n_models
        self.device = torch.device('cuda')
        torch.manual_seed(8734)        
        
        self.ensemble = [models.LeNet().to(self.device) for _ in range(self.n_models)]
        self.attach_optimizers()

        if self.dataset == 'mnist':
            self.data_train, self.data_test = datagen.load_mnist()
        elif self.dataset == 'cifar':
            self.data_train, self.data_test = datagen.load_cifar()

        self.best_test_acc = 0.
        self.best_test_loss = np.inf
        print (self.ensemble[0], ' X {}'.format(self.n_models))

    def attach_optimizers(self):
        self.optimizers = [torch.optim.Adam(m.parameters(), self.lr, weight_decay=self.wd)
                for m in self.ensemble]

    def train(self):
        for epoch in range(self.epochs):
            for batch_idx, (data, target) in enumerate(self.data_train):
                data, target = data.to(self.device), target.to(self.device)
                losses, corrects = [], []
                for i, model in enumerate(self.ensemble):
                    output = model(data)
                    loss = F.cross_entropy(output, target)
                    self.optimizers[i].zero_grad()
                    loss.backward()
                    self.optimizers[i].step()

                    losses.append(loss)
                    pred = output.data.max(1, keepdim=True)[1]
                    corrects.append(pred.eq(target.data.view_as(pred)).float().cpu().sum())

                loss = torch.stack(losses).mean().item()
                correct = torch.stack(corrects).mean().item()
                
                # Update Statistics
                if batch_idx % 100 == 0:
                    acc = (100* (correct / len(data)))
                    print ('--------------------------------------')
                    print ('{} train, epoch: {}'.format(args.dataset, epoch))
                    print ('Mean Acc: {}, Mean Loss: {}'.format(acc, loss))
                    print ('best test loss: {}'.format(self.best_test_loss))
                    print ('best test acc: {}'.format(self.best_test_acc))
                    print ('--------------------------------------')
            
            # Testing 
            with torch.no_grad():
                test_loss, test_acc, correct = self.test(voting=self.vote)
                print ('[Epoch {}] Ensemble [{}]. Test Loss: {}, Test Accuracy: {},  ({}/{})'.format(
                        epoch+1, self.n_models, test_loss, test_acc, correct, len(self.data_test.dataset)))
            
            if self.test_uncertainty:
                with torch.no_grad():
                    if self.dataset == 'mnist':
                        test_fn = uncertainty.eval_mnist_ensemble
                        plot_density = utils.plot_density_mnist
                    elif self.dataset == 'cifar':
                        test_fn = uncertainty.eval_cifar5_ensemble
                        plot_density = utils.plot_density_cifar
                    # inliers
                    entropy_i, variance_i = test_fn(self.ensemble)
                    # outliers
                    entropy_o, variance_o = test_fn(self.ensemble, outlier=True)
                    x_inliers = (entropy_i, variance_i)
                    x_outliers = (entropy_o, variance_o)
                    prefix = 'figures/ensemble/mnist/{}models'.format(self.n_models)
                    plot_density(x_inliers, x_outliers, self.n_models, prefix, epoch+1)

            if test_loss < self.best_test_loss or test_acc > self.best_test_acc:
                print ('==> new best stats, saving')
                if test_loss < self.best_test_loss:
                    self.best_test_loss = test_loss
                if test_acc > self.best_test_acc:
                    self.best_test_acc = test_acc


    def test(self, voting):
        test_acc = 0.
        test_loss = 0.
        total_correct = 0.
        
        for model in self.ensemble:
            model.eval()
        for i, (data, target) in enumerate(self.data_test):
            data = data.cuda()
            target = target.cuda()
            losses, outputs = [], []
            for model in self.ensemble:
                output = model(data)
                losses.append(F.cross_entropy(output, target))
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
        for model in self.ensemble:
            model.train()

        return test_loss, test_acc, total_correct


if __name__ == '__main__':
    args = load_args()
    trainer = EnsembleTrainer(args)
    trainer.train()


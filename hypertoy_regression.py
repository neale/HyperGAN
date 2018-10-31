import sys
import glob
import torch
import pprint
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly

from torch import optim
from torch.nn import functional as F

import models.models_toy as models

import ops
import utils
import netdef
import datagen


def load_args():

    parser = argparse.ArgumentParser(description='param-wgan')
    parser.add_argument('--z', default=128, type=int, help='latent space width')
    parser.add_argument('--ze', default=128, type=int, help='encoder dimension')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=200000, type=int)
    parser.add_argument('--target', default='small2', type=str)
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--beta', default=1000, type=int)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--use_x', default=False, type=bool)
    parser.add_argument('--pretrain_e', default=False, type=bool)
    parser.add_argument('--scratch', default=False, type=bool)
    parser.add_argument('--exp', default='0', type=str)
    parser.add_argument('--use_d', default=False, type=str)
    parser.add_argument('--model', default='small', type=str)

    args = parser.parse_args()
    return args


# hard code the two layer net
def train_clf(args, Z, data, target):
    """ calc classifier loss on target architecture """
    data, target = data.cuda(), target.cuda()
    data = data.view(20, 1)
    out = F.linear(data, Z[0])
    out = F.elu(out)
    out = F.linear(out, Z[1])
    out = out.view(20, 1)
    target = target.view(20, 1)
    loss_mse = F.mse_loss(out, target)
    pred = out.data.max(1, keepdim=True)[0]
    correct = pred.eq(target.data.view_as(pred)).cpu().sum()
    return (correct, out, loss_mse)


def f():
    x = np.random.uniform(-4, 4, (18))
    x = np.concatenate((x, [-4], [4]))
    eps = np.random.normal(0, 3, (20))
    y = np.power(x, 3) + eps
    return x, y


def plot(args, out, data, target):
    plt.figure()
    x_s = data.detach().view(-1).numpy()
    y_s = target.detach().view(-1).numpy()
    y_pred = out.detach().cpu().view(-1).numpy()
    x = np.arange(-6, 6.01, 0.01)
    y = np.power(x, 3)
    plt.plot(x, y)
    plt.scatter(x_s, y_s, c='r')
    coefs = poly.polyfit(x_s, y_pred, 3)
    x_new = np.linspace(x[0], x[-1], num=len(x)*10)
    ffit = poly.polyval(x_new, coefs)
    plt.plot(x_new, ffit, c='gray')
    plt.savefig('./images/toy_{}.png'.format(args.ze))
    plt.close()


def train(args):   
    netE = models.Encoder(args).cuda()
    W1 = models.GeneratorW1(args).cuda()
    W2 = models.GeneratorW2(args).cuda()
    print (netE, W1, W2)

    optimE = optim.Adam(netE.parameters(), lr=5e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimW1 = optim.Adam(W1.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimW2 = optim.Adam(W2.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    
    best_test_acc, best_test_loss = 0., np.inf
    args.best_loss, args.best_acc = best_test_loss, best_test_acc

    inputs, targets = f()
    inputs = torch.from_numpy(inputs); targets = torch.from_numpy(targets)
    dataset = torch.utils.data.TensorDataset(inputs.float(), targets.float())
    train = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True)

    x_dist = utils.create_d(args.ze)
    z_dist = utils.create_d(args.z)
    one = torch.FloatTensor([1]).cuda()
    mone = (one * -1).cuda()
    print ("==> pretraining encoder")
    j = 0
    final = 100.
    e_batch_size = 1000
    if args.pretrain_e:
        for j in range(100):
            x = utils.sample_d(x_dist, e_batch_size)
            z = utils.sample_d(z_dist, e_batch_size)
            codes = netE(x)
            for i, code in enumerate(codes):
                code = code.view(e_batch_size, args.z)
                mean_loss, cov_loss = ops.pretrain_loss(code, z)
                loss = mean_loss + cov_loss
                loss.backward(retain_graph=True)
            optimE.step()
            netE.zero_grad()
            print ('Pretrain Enc iter: {}, Mean Loss: {}, Cov Loss: {}'.format(
                j, mean_loss.item(), cov_loss.item()))
            final = loss.item()
            if loss.item() < 0.1:
                print ('Finished Pretraining Encoder')
                break

    best_loss = 100.
    print ('==> Begin Training')
    for epoch in range(args.epochs):
        for data, target in train:
            netE.zero_grad(); W1.zero_grad(); W2.zero_grad()
            z = utils.sample_d(x_dist, args.batch_size)
            codes = netE(z)
            l1 = W1(codes[0])
            l2 = W2(codes[1])
            for (g1, g2) in zip(l1, l2):
                correct, out, loss_mse = train_clf(args, [g1, g2], data, target)
                loss_mse.backward(retain_graph=True)
            optimE.step(); optimW1.step(); optimW2.step()
            loss = loss_mse.item()
            acc = correct / 20.   
            old_data = data
            old_target = target
        if epoch % 200 == 0:
            if loss_mse < best_loss:
                utils.save_hypernet_regression(args, [netE, W1, W2], loss_mse)
                plot(args, out, data, target)
            print ('**************************************')
            print ('Acc: {}, MSE Loss: {}'.format(acc, loss_mse))
            print ('**************************************')
            
        
if __name__ == '__main__':
    args = load_args()
    train(args)
    #load_toy(args)

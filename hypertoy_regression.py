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



def load_toy(args):
 
    path = '/scratch/eecs-share/ratzlafn/HyperGAN/exp_models/'
    paths = glob.glob(path+'*.pt')
    path = [x for x in paths if '15.77' in x][0]
    print ('loading path {}'.format(path))
    netE = models.Encoder(args)
    W1 = models.GeneratorW1(args)
    W2 = models.GeneratorW2(args)
    print (netE, W1, W2)
    d = torch.load(path)
    netE = utils.load_net_only(netE, d['E'])
    W1 = utils.load_net_only(W1, d['W1'])
    W2 = utils.load_net_only(W2, d['W2'])
    
    def clf(data, Z):
        data = data.view(20, 1)
        out = F.linear(data, Z[0])
        out = F.elu(out)
        out = F.linear(out, Z[1])
        return out.view(20, 1)

    def func(n):
        x = np.random.uniform(-n, n, (18))
        x = np.concatenate((x, [-n], [n]))
        eps = np.random.normal(0, 3, (20))
        y = np.power(x, 3) + eps
        return x, y
    
    def sample(x_s):
        z = torch.randn(args.batch_size, args.ze)
        codes = netE(z)
        l1 = W1(codes[0])
        l2 = W2(codes[1])
        outs = []
        for i in range(32):
            out = clf(torch.from_numpy(x_s).float(), [l1[i], l2[i]])
            outs.append(out)
        return torch.stack(outs)

    
    for pic in range(20):
        x_s, y_s = func(4)
        x = np.arange(-6, 6.01, 0.01)
        y = np.power(x, 3)

        x_new = np.linspace(x[0], x[-1], num=len(x)*10)
        x_i, y_i = func(6)
        outs = sample(x_s)
        outs_mean = outs.mean(0)
        outs_mean = outs_mean.detach().cpu().numpy()

        """ std max """
        out_full = []
        res_max = np.zeros(20)
        res_min = np.zeros(20)
        for _ in range(1000):
            outs = sample(x_i)
            outs = outs.detach().cpu().numpy()
            out_full.append(outs)
        outs = np.stack(out_full).reshape(1000*32, 20)
        for i in range(32*1000):
            for j in range(20):
                res_max[j] = outs[i][j] if outs[i][j] > res_max[j] else res_max[j]
                res_min[j] = outs[i][j] if outs[i][j] < res_min[j] else res_min[j]
        coefs1 = poly.polyfit(x_s, res_min, 3)
        ffit1 = poly.polyval(x_new, coefs1)
        coefs2 = poly.polyfit(x_s, res_max, 3)
        ffit2 = poly.polyval(x_new, coefs2)
        
        plt.plot(x_new, ffit1, color='#7CD6ED')
        plt.plot(x_new, ffit2, color='#7CD6ED')
        plt.fill_between(x_new, ffit1, ffit2, color='#C8E7EF')
        """
        outs = np.stack(out_full).reshape(200*32, 20)
        stds = outs.std(0)
        std1_max = outs.mean(0) + 2* np.std(outs, 0)
        std1_min = outs.mean(0) - 2* np.std(outs, 0)
        std2_max = outs.mean(0) + 3 * np.std(outs, 0)
        std2_min = outs.mean(0) - 3 * np.std(outs, 0)
        
        coefs1 = poly.polyfit(x_s, std2_min, 3)
        ffit1 = poly.polyval(x_new, coefs1)
        coefs2 = poly.polyfit(x_s, std2_max, 3)
        ffit2 = poly.polyval(x_new, coefs2)
        plt.plot(x_new, ffit1, color='red')
        plt.plot(x_new, ffit2, color='red')
        plt.fill_between(x_new, ffit1, ffit2, color='#7CD6ED')
        
        coefs1 = poly.polyfit(x_s, std1_min, 3)
        ffit1 = poly.polyval(x_new, coefs1)
        coefs2 = poly.polyfit(x_s, std1_max, 3)
        ffit2 = poly.polyval(x_new, coefs2)
        plt.plot(x_new, ffit1, color='green')
        plt.plot(x_new, ffit2, color='green')
        plt.fill_between(x_new, ffit1, ffit2, color='#31C6EC')
        """
        plt.plot(x, y, label='True Function')
        plt.scatter(x_s, y_s, color='red', label='Observations')
        coefs = poly.polyfit(x_s, outs_mean, 3)
        ffit = poly.polyval(x_new, coefs)
        plt.plot(x_new, ffit.reshape(-1), c='gray', label='Mean Function')
        plt.grid(True)
        plt.legend(loc='best')

        plt.savefig('/scratch/eecs-share/ratzlafn/mean_field_{}.png'.format(pic))
        plt.close()


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
    plt.savefig('/scratch/eecs-share/ratzlafn/toy_{}.png'.format(args.ze))
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
    load_toy(args)
    #train(args)

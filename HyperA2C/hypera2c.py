import torch
import torchvision
import torch.distributions.multivariate_normal as N

from torch import nn
from torch import autograd
from torch import optim
from torch.nn import functional as F
from torchvision import datasets, transforms

import numpy as np

import netdef
import utils_xai as utils


def sample_x(args, gen, id, grad=True):
    if type(gen) is list:
        res = []
        for i, g in enumerate(gen):
            data = next(g)
            x = (torch.tensor(data, requires_grad=grad)).cuda()
            res.append(x.view(*args.shapes[i]))
    else:
        data = next(gen)
        x = torch.tensor(data, requires_grad=grad).cuda()
        res = x.view(*args.shapes[id])
    return res


def sample_z(args, grad=True):
    z = torch.randn(args.batch_size, args.dim, requires_grad=grad).cuda()
    return z


def sample_z_like(shape, scale=1., grad=True):
    mean = torch.zeros(shape[1])
    cov = torch.eye(shape[1])
    D = N.MultivariateNormal(mean, cov)
    z = D.sample((shape[0],)).cuda()
    return scale * z


def batch_zero_grad(nets):
    for module in nets:
        module.zero_grad()


def batch_update_optim(optimizers):
    for optim in optimizers:
        optim.step()

def free_params(nets):
    for module in nets:
        for p in module.parameters():
            p.requires_grad = True


def frozen_params(nets):
    for module in nets:
        for p in module.parameters():
            p.requires_grad = False


def pretrain_loss(encoded, noise):
    mean_z = torch.mean(noise, dim=0, keepdim=True)
    mean_e = torch.mean(encoded, dim=0, keepdim=True)
    mean_loss = F.mse_loss(mean_z, mean_e)

    cov_z = torch.matmul((noise-mean_z).transpose(0, 1), noise-mean_z)
    cov_z /= 999
    cov_e = torch.matmul((encoded-mean_e).transpose(0, 1), encoded-mean_e)
    cov_e /= 999
    cov_loss = F.mse_loss(cov_z, cov_e)
    return mean_loss, cov_loss


def pretrain_encoder(args, E, Optim):

    j = 0
    final = 100.
    e_batch_size = 2000
    for j in range(3000):
        x = sample_z_like((e_batch_size, args.ze))
        z = sample_z_like((e_batch_size, args.z))
        codes = E(x)
        for i, code in enumerate(codes):
            code = code.view(e_batch_size, args.z)
            mean_loss, cov_loss = pretrain_loss(code, z)
            loss = mean_loss + cov_loss
            loss.backward(retain_graph=True)
        Optim['optimE'].step()
        E.zero_grad()
        Optim['optimE'].zero_grad()
        print ('Pretrain Enc iter: {}, Mean Loss: {}, Cov Loss: {}'.format(
            j, mean_loss.item(), cov_loss.item()))
        final = loss.item()
        if loss.item() < 0.1:
            print ('Finished Pretraining Encoder')
            break
    return E, Optim


def get_policy_weights(args, HyperNet, Optim):
    # generate embedding for each layer
    batch_zero_grad([HyperNet.encoder] + HyperNet.generators)
    z = sample_z_like((args.batch_size, args.ze))
    codes = HyperNet.encoder(z)
    layers = []
    # decompress to full parameter shape
    for (code, gen) in zip(codes, HyperNet.generators):
        layers.append(gen(code).mean(0))
    # Z Adversary 
    free_params([HyperNet.adversary])
    frozen_params([HyperNet.encoder] + HyperNet.generators)
    for code in codes:
        noise = sample_z_like((args.batch_size, args.z))
        d_real = HyperNet.adversary(noise)
        d_fake = HyperNet.adversary(code.contiguous())
        d_real_loss = -1 * torch.log((1-d_real).mean())
        d_fake_loss = -1 * torch.log(d_fake.mean())
        d_real_loss.backward(retain_graph=True)
        d_fake_loss.backward(retain_graph=True)
        d_loss = d_real_loss + d_fake_loss
    Optim['optimD'].step()
    free_params([HyperNet.encoder] + HyperNet.generators)
    frozen_params([HyperNet.adversary])

    return layers, HyperNet, Optim


def update_hn(args, loss, HyperNet, Optim):

    scaled_loss = (args.beta*loss) #+ z1_loss + z2_loss + z3_loss
    scaled_loss.backward()
    Optim['optimE'].step()
    Optim['optimG'][0].step()
    torch.nn.utils.clip_grad_norm_(HyperNet.generators[0].parameters(), 40)
    Optim['optimG'][1].step()
    torch.nn.utils.clip_grad_norm_(HyperNet.generators[1].parameters(), 40)
    Optim['optimG'][2].step()
    torch.nn.utils.clip_grad_norm_(HyperNet.generators[2].parameters(), 40)
    Optim['optimG'][3].step()
    torch.nn.utils.clip_grad_norm_(HyperNet.generators[3].parameters(), 40)
    Optim['optimG'][4].step()
    torch.nn.utils.clip_grad_norm_(HyperNet.generators[4].parameters(), 40)
    Optim['optimG'][5].step()
    torch.nn.utils.clip_grad_norm_(HyperNet.generators[5].parameters(), 40)
    Optim['optimE'].zero_grad()
    Optim['optimG'][0].zero_grad()
    Optim['optimG'][1].zero_grad()
    Optim['optimG'][2].zero_grad()
    Optim['optimG'][3].zero_grad()
    Optim['optimG'][4].zero_grad()
    Optim['optimG'][5].zero_grad()


    return HyperNet, Optim


def update(args):
    # Update Statistics """
    if batch_idx % 50 == 0:
        acc = (correct / 1) 
        print ('**************************************')
        print ('Enc, Dz, Lscale: {} test'.format(args.beta))
        print ('Acc: {}, G Loss: {}, D Loss: {}'.format(acc, loss, d_loss))
        print ('best test loss: {}'.format(args.best_loss))
        print ('best test acc: {}'.format(args.best_acc))
        print ('**************************************')
    if batch_idx % 100 == 0:
        test_acc = 0.
        test_loss = 0.
        for i, (data, y) in enumerate(cifar_test):
            z = sample_z_like((args.batch_size, args.ze,))
            l_out = []
            codes = netE(z)
            for code, gen in zip(codes, generators):
                l_out.append(gen(code))
            min_loss_batch = 10.
            z_test = [l1[0], l2[0], l3[0], l4[0], l5[0]]
            for (z1, z2, z3, z4, z5) in zip(*l_out):
                correct, loss = train_clf(args, [z1, z2, z3, z4, z5], data, y, val=True)
                if loss.item() < min_loss_batch:
                    min_loss_batch = loss.item()
                    z_test = [z1, z2, z3, z4, z5]
                test_acc += correct.item()
                test_loss += loss.item()
        #y_acc, y_loss = utils.test_samples(args, z_test, train=True)
        test_loss /= len(cifar_test.dataset) * 32
        test_acc /= len(cifar_test.dataset) * 32
        print ('Test Accuracy: {}, Test Loss: {}'.format(test_acc, test_loss))
        # print ('FC Accuracy: {}, FC Loss: {}'.format(y_acc, y_loss))
        if test_loss < best_test_loss or test_acc > best_test_acc:
            print ('==> new best stats, saving')
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                args.best_loss = test_loss
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                args.best_acc = test_acc

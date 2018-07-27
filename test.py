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
from torchvision import datasets, transforms
from torch import nn
from torch import autograd
from torch import optim
from torch.nn import functional as F
import pprint

import ops
import plot
import utils
import netdef
import datagen
import matplotlib.pyplot as plt


def load_args():

    parser = argparse.ArgumentParser(description='param-wgan')
    parser.add_argument('-z', '--dim', default=64, type=int, help='latent space width')
    parser.add_argument('-g', '--gp', default=10, type=int, help='gradient penalty')
    parser.add_argument('-b', '--batch_size', default=64, type=int)
    parser.add_argument('-e', '--epochs', default=200000, type=int)
    parser.add_argument('-s', '--model', default='small', type=str)
    parser.add_argument('-d', '--dataset', default='mnist', type=str)
    parser.add_argument('-l', '--layer', default='all', type=str)
    parser.add_argument('-zd', '--depth', default=2, type=int, help='latent space depth')
    parser.add_argument('--nfe', default=64, type=int)
    parser.add_argument('--nfg', default=64, type=int)
    parser.add_argument('--nfd', default=128, type=int)
    parser.add_argument('--beta', default=1000, type=int)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--comet', default=False, type=bool)
    parser.add_argument('--use_wae', default=True, type=bool)
    parser.add_argument('--val_iters', default=10, type=bool)

    args = parser.parse_args()
    return args


class Encoder_fc(nn.Module):
    def __init__(self, args):
        super(Encoder_fc, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'Encoder'
        self.linear0 = nn.Linear(self.lcd*self.lcd*10, self.lcd*self.lcd)
        self.linear1 = nn.Linear(self.lcd*self.lcd, self.nfe*4)
        self.linear2 = nn.Linear(self.nfe*4, self.nfe*2)
        self.linear3 = nn.Linear(self.nfe*2, self.dim)
        self.relu = nn.ELU(inplace=True)
        self.bn0 = nn.BatchNorm2d(self.lcd*self.lcd)
        self.bn1 = nn.BatchNorm2d(self.nfe*4)
        self.bn2 = nn.BatchNorm2d(self.nfe*2)
        self.bn3 = nn.BatchNorm2d(self.dim)
        #self.relu = nn.LeakyReLU(.2, inplace=True)

    def forward(self, x):
        #print ('E in: ', x.shape)
        x = x.view(self.batch_size, -1) #flatten filter size
        if x.shape[-1] > self.lcd*self.lcd:
            x = self.relu(self.linear0(x))
            # x = self.relu(self.bn0(self.linear0(x)))
        if self.use_wae:
            z = torch.normal(torch.zeros_like(x.data), std=0.01)
            x.data += z
        
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        """
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
       
        """
        #print ('E out: ', x.shape)
        return x


class Generator_fc(nn.Module):
    def __init__(self, args):
        super(Generator_fc, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'Generator'
        self.linear1 = nn.Linear(self.dim, self.nfg*4)
        self.linear2 = nn.Linear(self.nfg*4, self.nfg*2)
        self.linear3 = nn.Linear(self.nfg*2, self.nfg)
        self.linear_out = nn.Linear(self.nfg, self.lcd*self.lcd)
        self.relu = nn.LeakyReLU(.2, inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # print ('G in: ', x.shape)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        z = torch.normal(torch.zeros_like(x.data), std=0.01)
        x.data += z
        x = self.linear_out(x)
        x = x.view(-1, self.lcd, self.lcd)
        # print ('G out: ', x.shape)
        return x


class Generator_conv(nn.Module):
    def __init__(self, args):
        super(Generator_conv, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)

        self.name = 'Generator'
        self.linear = nn.Linear(self.nfg, 64*64)
        self.conv1 = nn.Conv2d(1, self.nfg, 3, 2)
        self.conv2 = nn.Conv2d(self.nfg, self.nfg*2, 3, 2)
        self.conv3 = nn.Conv2d(self.nfg*2, self.nfg*4, 3, 2)
        self.relu = nn.LeakyReLU(.2, inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        print ('G in: ', x.shape)
        x = self.elu(self.linear(x))
        x = x.view(-1, 1, 64, 64)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        # x = self.tanh(self.conv3(x))
        x = x.view(-1, 128, 7, 7)
        print ('G out: ', x.shape)
        return x



class Discriminator_fc(nn.Module):
    def __init__(self, args):
        super(Discriminator_fc, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)

        self.name = 'Discriminator'
        self.ng = self.lcd * self.lcd
        self.linear1 = nn.Linear(self.ng, self.nfd)
        self.linear2 = nn.Linear(self.nfd, self.nfd)
        self.linear3 = nn.Linear(self.nfd, 1)
        self.relu = nn.LeakyReLU(.2, inplace=True)

    def forward(self, x):
        # print ('D in: ', x.shape)
        x = x.view(-1, self.ng)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        # print ('D out: ', x.shape)
        return x

class Discriminator_wae(nn.Module):
    def __init__(self, args):
        super(Discriminator_wae, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        
        self.name = 'Discriminator_wae'
        self.linear1 = nn.Linear(self.dim, 256)
        self.linear2 = nn.Linear(256, 1)
        self.relu = nn.LeakyReLU(.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print ('Dz in: ', x.shape)
        # x = x.view(-1, self.ng)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.sigmoid(x)
        # print ('Dz out: ', x.shape)
        return x


def calc_gradient_penalty(args, netD, real_data, gen_data):
    batch_size = args.batch_size
    datashape = args.shapes[args.id]
    alpha = torch.rand(datashape[0], 1)
    alpha = alpha.expand(datashape[0], real_data.nelement()//datashape[0])
    alpha = alpha.contiguous().view(*datashape).cuda()
    interpolates = alpha * real_data + ((1 - alpha) * gen_data).cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.gp
    return gradient_penalty


def sample_x(args, gen, id):
    if type(gen) is list:
        res = []
        for i, g in enumerate(gen):
            data = next(g)
            x = autograd.Variable(torch.Tensor(data)).cuda()
            res.append(x.view(*args.shapes[i]))
    else:
        data = next(gen)
        x = torch.Tensor(data).cuda()
        x = x.view(*args.shapes[id])
        res = autograd.Variable(x)
    return res


def sample_z(args):
    z = torch.randn(args.batch_size, args.dim).cuda()
    z = autograd.Variable(z)
    return z
 

def train_ae(args, netG, netE, x):
    netG.zero_grad()
    netE.zero_grad()
    x_encoding = netE(x)
    x_fake = ops.gen_layer(args, netG, x_encoding)
    # fake = netG(encoding)
    x_fake = x_fake.view(*args.shapes[args.id])
    ae_loss = F.mse_loss(x_fake, x)
    return ae_loss, x_fake


def train_wadv(args, netDz, netE, x, z):
    netDz.zero_grad()
    z_fake = netE(x).view(args.batch_size, -1)
    Dz_real = netDz(z)
    Dz_fake = netDz(z_fake)
    Dz_loss = -(torch.mean(Dz_real) - torch.mean(Dz_fake))
    Dz_loss.backward()


def train_adv(args, netD, x, z):
    netD.zero_grad()
    D_real = netD(x).mean()
    z = z.view(*args.shapes[args.id])
    # fake = netG(z)
    D_fake = netD(z).mean()
    gradient_penalty = calc_gradient_penalty(args, netD,
            x.data, z.data)
    return D_real, D_fake, gradient_penalty


def load_mnist():
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                batch_size=64, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=64, shuffle=True, **kwargs)
    return train_loader, test_loader


# hard code the two layer net
def train_clf(args, Z, data, target, val=False):
    """ calc classifier loss """
    data = autograd.Variable(data).cuda(),
    target = autograd.Variable(target).cuda()
    #im = data[0][0, 0, :, :].cpu().data.numpy()
    #import matplotlib.pyplot as plt
    #plt.imshow(im, cmap='gray')
    #plt.show()
    out = F.conv2d(data[0], Z[0], padding=4)
    out = F.elu(out)
    out = F.max_pool2d(out, 4, 4)
    out = out.view(-1, 3136)
    out = F.linear(out, Z[1])
    loss = F.cross_entropy(out, target)
    correct = None
    if val:
        pred = out.data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).long().cpu().sum()
    # (clf_acc, clf_loss), (oa, ol) = ops.clf_loss(args, Z)
    return (correct, loss)


def train_gen(args, netG, netD):
    netG.zero_grad()
    z = sample_z(args)
    fake = netG(z)
    G = netD(fake).mean()
    G_cost = -G
    return G

def ae_code():

    print ("==> pretraining Autoencoder")
    for i in range(0):
        ae_losses, layers, samples = [], [], []
        # lets hardcode then optimize
        args.id = 0
        netG.zero_grad()
        netE.zero_grad()
        x = sample_x(args, param_gen[0], 0) # sample
        x_enc = netE(x)
        x_fake = ops.gen_layer(args, netG, x_enc)
        x_fake = x_fake.view(*args.shapes[0])
        ae_loss = F.mse_loss(x_fake, x)
        ae_loss.backward(retain_graph=True)
        ae_losses.append(ae_loss.cpu().data.numpy()[0])
        optimizerE.step()
        optimizerG.step()

        args.id += 1
        netG.zero_grad()
        netE.zero_grad()
        xl = x_fake
        xl = xl.view(-1, *args.shapes[1][1:])
        xl_target = sample_x(args, param_gen[1], 1)
        xl_enc = netE(xl)
        xl_fake = ops.gen_layer(args, netG, xl_enc)
        xl_fake = xl_fake.view(*args.shapes[1])
        ae_loss = F.mse_loss(xl_fake, xl_target)
        ae_loss.backward()
        ae_losses.append(ae_loss.cpu().data.numpy()[0])
        optimizerE.step()
        optimizerG.step()

        if i % 500 == 0:
            norm_x = np.linalg.norm(x.data)
            norm_z = np.linalg.norm(x_fake.data)

            norm_xl = np.linalg.norm(xl_target.data)
            norm_zl = np.linalg.norm(xl_fake.data)

            cov_x_z = cov(x, x_fake).data[0]
            cov_xl_zl = cov(xl_target, xl_fake).data[0]
            print (ae_losses, 'CONV-- G: ', norm_z, '-->', norm_x, 
                    'LINEAR-- G: ', norm_zl, '-->', norm_xl)
            """
            utils.plot_histogram([x.cpu().data.numpy().flatten(),
                                  x_fake.cpu().data.numpy().flatten()],
                                  save=False, id='conv iter {}'.format(i))
            utils.plot_histogram([xl_target.cpu().data.numpy().flatten(),
                                  xl_fake.cpu().data.numpy().flatten()],
                                  save=False, id='linear iter {}'.format(i))
            """
def gan_train_code():

    for iteration in range(0, args.epochs):
        start_time = time.time()

        """ Update AE """
        # print ("==> autoencoding layers")
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation

        print ('==> updating AE') 
        for batch_idx, (data, target) in enumerate(mnist_train):
            ae_losses = []
            netE.zero_grad()
            netG.zero_grad()
            args.id = 0  # reset
            x = sample_x(args, param_gen[0], 0)
            z1 = ops.gen_layer(args, netG, netE(x))
            z1 = z1.view(*args.shapes[0])
            z1_loss = F.mse_loss(z1, x)
            args.id = 1
            x2 = sample_x(args, param_gen[1], 1)
            z2 = z1.view(-1, *args.shapes[1][1:])
            z2 = ops.gen_layer(args, netG, netE(z2))
            z2 = z2.view(*args.shapes[1])
            z2_loss = F.mse_loss(z2, x2)
            correct, loss = train_clf(args, [z1, z2], data, target, val=True)
            scaled_loss = (loss*.05) + z2_loss + z1_loss
            scaled_loss.backward(retain_graph=True)
            optimizerE.step()
            optimizerG.step()
            ae_losses.append(z1_loss.cpu().data.numpy()[0])
            ae_losses.append(z2_loss.cpu().data.numpy()[0])
            clf_loss = loss.cpu().data.numpy()[0]
            acc = correct / (float(len(target)))

            # Update Adversary 
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            # print ('==> updating D')
            layers, d_losses, w1_losses = [], [], []
            args.id = 0  # reset
            x = sample_x(args, param_gen[0], id=0)
            z1 = ops.gen_layer(args, netG, netE(x))
            z1 = z1.view(*args.shapes[0])
            d_real, d_fake, gp = train_adv(args, netD, x, z1)
            d_real.backward(torch.Tensor([-1]).cuda(), retain_graph=True)
            d_fake.backward(retain_graph=True)
            gp.backward()
            optimizerD.step()
            w1_losses.append((d_real - d_fake).cpu().data.numpy()[0])
            d_losses.append((d_fake - d_real + gp).cpu().data.numpy()[0])
            layers.append(z1)

            args.id = 1
            x2 = sample_x(args, param_gen[1], 1)
            z2 = z1.view(-1, *args.shapes[1][1:])
            z2 = ops.gen_layer(args, netG, netE(z2))
            z2 = z2.view(*args.shapes[1])
            d_real, d_fake, gp = train_adv(args, netD, x2, z2)
            d_real.backward(torch.Tensor([-1]).cuda(), retain_graph=True)
            d_fake.backward(retain_graph=True)
            gp.backward()
            optimizerD.step()
            w1_losses.append((d_real - d_fake).cpu().data.numpy()[0])
            d_losses.append((d_fake - d_real + gp).cpu().data.numpy()[0])
            layers.append(z2)

            # correct, loss = train_clf(args, layers, data, target, val=True)
            # loss.backward()
            # optimizerD.step()

            # print ("==> updating g")
            g_losses = []
            args.id = 0
            g_cost = train_gen(args, netG, netD)
            g_cost.backward(torch.Tensor([-1]).cuda())
            g_losses.append(g_cost.cpu().data.numpy()[0])
            optimizerG.step()
            args.id = 1
            g_cost = train_gen(args, netG, netD)
            g_cost.backward(torch.Tensor([-1]).cuda())
            g_losses.append(g_cost.cpu().data.numpy()[0])
            optimizerG.step()

            # Write logs
            if batch_idx % 100 == 0:
                print ('==> iter: ', iteration)
                print('AE cost', ae_losses)
                # save_dir = './plots/{}/{}'.format(args.dataset, args.model)
                # path = 'params/sampled/{}/{}'.format(args.dataset, args.model)
                # utils.save_model(args, netE, optimizerE)
                # utils.save_model(args, netG, optimizerG)
                # utils.save_model(args, netD, optimizerD)
                # print ("==> saved model instances")
                # if not os.path.exists(path):
                #     os.makedirs(path)
                # samples = netG(z)
                print ("****************")
                print('Iter ', batch_idx, 'Beta ', args.beta)
                print('D cost', d_losses)
                print('G cost', g_losses)
                print('AE cost', ae_losses)
                print('W1 distance', w1_losses)
                print ('clf (acc)', acc)
                print ('clf (loss', clf_loss)
                # print ('filter 1: ', layers[0][0, 0, :, :], layers[1][:, 0])
                print ("****************")

def cov(x, y):
    mean_x = torch.mean(x, dim=0, keepdim=True)
    mean_y = torch.mean(y, dim=0, keepdim=True)
    cov_x = torch.matmul((x-mean_x).transpose(0, 1), x-mean_x)
    cov_x /= 999
    cov_y = torch.matmul((y-mean_y).transpose(0, 1), y-mean_y)
    cov_y /= 999
    cov_loss = F.mse_loss(cov_y, cov_x)
    return cov_loss


def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def train(args):
    
    torch.manual_seed(1)
    
    netE = Encoder_fc(args).cuda()
    netG = Generator_fc(args).cuda()
    netD = Discriminator_fc(args).cuda()
    print (netE, netG, netD)

    optimizerE = optim.Adam(netE.parameters(), lr=1e-3, betas=(0.5, 0.9), weight_decay=1e-4)
    optimizerG = optim.Adam(netG.parameters(), lr=1e-3, betas=(0.5, 0.9), weight_decay=1e-4)
    optimizerD = optim.Adam(netD.parameters(), lr=1e-3, betas=(0.5, 0.9), weight_decay=1e-4)
    
    if args.use_wae:
        netDz = Discriminator_wae(args).cuda()
        optimizerDz = optim.Adam(netDz.parameters(), lr=1e-3, betas=(0.5, 0.9))

    """
    base_gen = []
    param_gen = []
    base_gen.append((utils.dataset_iterator(args, 0)))
    base_gen.append((utils.dataset_iterator(args, 1)))
    param_gen.append(utils.inf_train_gen(base_gen[0][0]))
    param_gen.append(utils.inf_train_gen(base_gen[1][0]))
    """
    mnist_train, mnist_test = load_mnist()
    base_gen = datagen.load(args)
    conv_gen = utils.inf_train_gen(base_gen[0])
    linear_gen = utils.inf_train_gen(base_gen[1])
    acc = 0
    for batch_idx, (data, target) in enumerate(mnist_test):
        conv = autograd.Variable(torch.Tensor(next(conv_gen))).cuda()
        linear = autograd.Variable(torch.Tensor(next(linear_gen))).cuda()
        c, l = train_clf(args, [conv, linear], data, target, val=True)
        acc += c
    print (float(acc) / len(mnist_test.dataset) * 100, '%')

    print ('==> running clf experiments')
    for _ in range(1000):
        for batch_idx, (data, target) in enumerate(mnist_train):

            netE.zero_grad()
            netG.zero_grad()
            netDz.zero_grad()
            
            args.id = 0  # reset
            x, x2 = sample_x(args, [conv_gen, linear_gen], 0)
            z1 = ops.gen_layer(args, netG, netE(x))
            z1 = z1.view(*args.shapes[0])
            z1_loss = F.mse_loss(z1, x)
           
            z_real = sample_z(args)
            z_fake = netE(x)
            d_real = netDz(z_real)
            d_fake = netDz(z_fake)
            d_loss1 = -(torch.mean(d_real) - torch.mean(d_fake))
            d_loss1.backward()
            optimizerDz.step()
            for p in netDz.parameters():
                p.data.clamp_(-0.01, 0.01)

            args.id = 1
            z2 = z1.view(-1, *args.shapes[1][1:])
            z2 = ops.gen_layer(args, netG, netE(z2))
            z2 = z2.view(*args.shapes[1])
            z2_loss = F.mse_loss(z2, x2)

            netDz.zero_grad()
            z_real = sample_z(args)
            z_fake = netE(x2)
            d_real = netDz(z_real)
            d_fake = netDz(z_fake)
            d_loss2 = -(torch.mean(d_real) - torch.mean(d_fake))
            d_loss2.backward()
            optimizerDz.step()
            for p in netDz.parameters():
                p.data.clamp_(-0.01, 0.01)
            
            correct, loss = train_clf(args, [z1, z2], data, target, val=True)
            scaled_loss = (loss*.01) + z2_loss + z1_loss
            scaled_loss.backward(retain_graph=True)
            optimizerE.step()
            optimizerG.step()
            loss = loss.cpu().data.numpy()[0]
           
            if batch_idx % 50 == 0:
                acc = (correct / 1) 
                norm_x = np.linalg.norm(x.data)
                norm_z1 = np.linalg.norm(z1.data)
                norm_z2 = np.linalg.norm(z2.data)
                print (acc, loss, 'CONV-- G: ', norm_z1, '-->', norm_x,
                        'LINEAR-- G: ', norm_z2, 
                        'Dz -- ', d_loss1.cpu().data[0], d_loss2.cpu().data[0])

                plt.ion()
                fig, (ax, ax1) = plt.subplots(1, 2)
                x = [x.cpu().data.numpy().flatten(), z1.cpu().data.numpy().flatten()]
                for i in range(len(x)):
                    n, bins, patches = ax.hist(x[i], 50, density=True, alpha=0.75, label=str(i))
                ax.legend(loc='upper right')
                ax.set_title('conv1')
                ax.grid(True)
                #utils.plot_histogram([x.cpu().data.numpy().flatten(),
                #                      z1.cpu().data.numpy().flatten()],
                #                      save=True, id='conv iter {}'.format(batch_idx))
                # plt.ion()
                y = [x2.cpu().data.numpy().flatten(), z2.cpu().data.numpy().flatten()]
                for i in range(len(y)):
                    n, bins, patches = ax1.hist(y[i], 50, density=True, alpha=0.75, label=str(i))
                ax1.legend(loc='upper right')
                ax1.set_title('linear')
                ax1.grid(True)

                plt.draw()
                plt.pause(1.0)
                plt.close()


                """
                utils.plot_histogram([x2.cpu().data.numpy().flatten(),
                                      z2.cpu().data.numpy().flatten()],
                                      save=False, id='linear iter {}'.format(i))
                """

if __name__ == '__main__':

    args = load_args()
    modeldef = netdef.nets()[args.model]
    pprint.pprint (modeldef)
    # log some of the netstat quantities so we don't subscript everywhere
    args.stat = modeldef
    args.shapes = modeldef['shapes']
    args.lcd = modeldef['base_shape']
    # why is a running product so hard in python
    args.gcd = int(np.prod([*args.shapes[0]]))
    train(args)

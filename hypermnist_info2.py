import sys
import glob
import torch
import pprint
import argparse
import numpy as np
import itertools
from torch import nn

from torch import optim
from torch.nn import functional as F

import ops
import utils
import netdef
import datagen
import models.models_mnist_info as models


def load_args():

    parser = argparse.ArgumentParser(description='param-wgan')
    parser.add_argument('--z', default=100, type=int, help='latent space width')
    parser.add_argument('--ze', default=300, type=int, help='encoder dimension')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=200000, type=int)
    parser.add_argument('--target', default='small2', type=str)
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--beta', default=1, type=int)
    parser.add_argument('--alpha', default=1, type=int)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--use_x', default=False, type=bool)
    parser.add_argument('--load_e', default=False, type=bool)
    parser.add_argument('--pretrain_e', default=False, type=bool)
    parser.add_argument('--scratch', default=False, type=bool)
    parser.add_argument('--exp', default='0', type=str)
    parser.add_argument('--use_d', default=False, type=str)
    parser.add_argument('--model', default='info', type=str)
    parser.add_argument('--disc_iters', default=5, type=int)
    parser.add_argument('--factors', default=2, type=int)

    args = parser.parse_args()
    return args

""" conditional distribution Q(x|c) """
class Q(nn.Module):
    def __init__(self):
        super(Q, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(8*2*2, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 8*2*2)
        x = self.fc(x)
        return x


# hard code the two layer net
def train_clf(args, Z, data, target):
    """ calc classifier loss """
    data, target = data.cuda(), target.cuda()
    out = F.conv2d(data, Z[0], stride=1)
    out = F.leaky_relu(out)
    out = F.max_pool2d(out, 2, 2)
    out = F.conv2d(out, Z[1], stride=1)
    out = F.leaky_relu(out)
    out = F.max_pool2d(out, 2, 2)
    out = out.view(-1, 512)
    out = F.linear(out, Z[2])
    loss = F.cross_entropy(out, target)
    pred = out.data.max(1, keepdim=True)[1]
    correct = pred.eq(target.data.view_as(pred)).long().cpu().sum()
    return (correct, loss)


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def sample_categorical(shape):
    c = np.random.multinomial(1, 10*[.1], size=shape)
    c = torch.tensor(c, dtype=torch.float32).cuda()
    return c


def to_categorical(y, cols):
    y_cat = np.zeros((y.shape[0], cols))
    y_cat[range(y.shape[0]), y] = 1
    return torch.tensor(y_cat, dtype=torch.float).cuda()


def MI_loss(args, y, y_pred, c, c_pred):
    pred = y_pred.max(1, keepdim=True)[1]
    acc = pred.eq(y.data.view_as(pred)).long().cpu().sum()
    categorical_loss = F.cross_entropy(y_pred, y)
    continuous_loss = F.mse_loss(c_pred, c)
    mi_loss = categorical_loss + (.1 * continuous_loss)
    return mi_loss, (acc, categorical_loss)


def embedding_clf(args, layer, netQ, c):
    out = netQ(layer, clf=True)
    target = torch.tensor([torch.max(i, 0)[1].item() for i in c]).cuda()
    loss = F.cross_entropy(out, target.long())
    pred = out.data.max(1, keepdim=True)[1]
    acc = pred.eq(target.data.view_as(pred)).long().cpu().sum()
    return acc, loss


def x_gen(gen=None):
    path = '/scratch/eecs-share/ratzlafn/HyperGAN/conv_x/'
    files = glob.glob(path+'*.npy')
    params = []
    for file in files:
        params.append(np.load(file))
    return itertools.cycle(params)


def sample_layer(args, netE, W2, rows):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Static sample
    static_y = to_categorical(np.array([num for _ in range(10)
        for num in range(10)])[:32], cols=10)
    static_z = torch.zeros(args.batch_size, args.ze).cuda()
    static_c = torch.zeros((args.batch_size, args.factors)).cuda()
    static_netcode = netE(static_z)[1]

    z = torch.randn((args.batch_size, args.ze)).cuda()
    netcode = netE(z)[1]

    static_sample = W2(netcode, static_y, static_c)

    # Get varied c1 and c2
    zeros = np.zeros((32, 1))
    c_varied = np.repeat(np.linspace(-1, 1, 10)[:, np.newaxis], 3, 0)
    c_varied = np.append([[0, 0]], c_varied)[:, np.newaxis]
    c1 = torch.tensor(np.concatenate((c_varied, zeros), -1)).float().cuda()
    c2 = torch.tensor(np.concatenate((zeros, c_varied), -1)).float().cuda()
    sample1 = W2(static_netcode, static_y, c1)[np.random.randint(10)]
    sample2 = W2(static_netcode, static_y, c2)[np.random.randint(10)]
    return sample1, sample2


def train(args):
    
    torch.manual_seed(8734)
    netE = models.Encoder(args).cuda()
    W1 = models.GeneratorW1(args).cuda()
    W2 = models.GeneratorW2(args).cuda()
    W3 = models.GeneratorW3(args).cuda()
    netD = models.DiscriminatorQ(args).cuda()
    netQ = Q().cuda()
    print (netE, W1, W2, W3, netD, netQ)

    netD.apply(weight_init)

    optimE = optim.Adam(netE.parameters(), lr=5e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimW1 = optim.Adam(W1.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-4)
    optimW2 = optim.Adam(W2.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-4)
    optimW3 = optim.Adam(W3.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-4)
    optimD = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-4)

    #q_params = itertools.chain(W2.parameters(), netD.parameters())
    q_params = itertools.chain(W2.parameters(), netQ.parameters())
    optimQ = optim.Adam(q_params, lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-4)
    
    best_test_acc, best_test_loss = 0., np.inf
    args.best_loss, args.best_acc = best_test_loss, best_test_acc

    mnist_train, mnist_test = datagen.load_mnist(args)
    real_filters = x_gen()
    x_dist = utils.create_d(args.ze)
    z_dist = utils.create_d(args.z)
    one = torch.FloatTensor([1]).cuda()
    mone = (one * -1).cuda()
   
    if args.pretrain_e:
        j = 0
        final = 100.
        e_batch_size = 1000
        print ("==> pretraining encoder")
        for j in range(300):
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
    c_acc, c_loss = [], []
    print ('==> Begin Training')
    for _ in range(args.epochs):
        for batch_idx, (data, target) in enumerate(mnist_train):
            
            """ generate encoding """
            valid = torch.ones((args.batch_size, 1), dtype=torch.float32, requires_grad=False).cuda()
            fake = torch.zeros((args.batch_size, 1), dtype=torch.float32, requires_grad=False).cuda()
            real = torch.tensor(next(real_filters), requires_grad=True).cuda()
            labels = to_categorical(target.numpy(), cols=10)

            ops.batch_zero_grad([optimE, optimW1, optimW2, optimW3])
            z = utils.sample_d(x_dist, args.batch_size)
            c = torch.tensor(np.random.uniform(-1, 1, (args.batch_size, args.factors))).float().cuda()
            ycat = torch.tensor(np.random.randint(0, 10, args.batch_size)).long().cuda()
            y = to_categorical(ycat, cols=10).float()
            codes = netE(z)

            """ train generator """
            l1 = W1(codes[0])
            l2 = W2(codes[1], y)#, c)
            l3 = W3(codes[2]) 

            clf_loss = []
            for i, (g1, g2, g3) in enumerate(zip(l1, l2, l3)):
                #d_valid, d_y, d_f = netD(g2)
                correct, loss = train_clf(args, [g1, g2, g3], data, target)
                clf_loss.append(loss)
                #adv_loss = F.mse_loss(d_valid, valid)
                scaled_loss = args.beta * loss# + adv_loss
                scaled_loss.backward(retain_graph=True)
            
            optimE.step()
            optimW1.step()
            optimW2.step()
            optimW3.step()
            loss = torch.stack(clf_loss).mean().item()
            
            """ train discriminator """
            """ 
            optimD.zero_grad()
            for g2 in l2:
                real_pred, _, _ = netD(real)
                d_real_loss = F.mse_loss(real_pred, valid)
                fake_pred, _, _ = netD(g2.detach())
                d_fake_loss = F.mse_loss(fake_pred, fake)
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward(retain_graph=True)    
            optimD.step()        
            """
            """ MI loss """
            # want to maximize the mutaul information between the labels and a given filter
            # last conv layer only
            optimQ.zero_grad()
            optimE.zero_grad()
            sampled_labels = np.random.randint(0, 10, args.batch_size)
            gt_labels = torch.tensor(sampled_labels, requires_grad=False).long().cuda()
            label = to_categorical(sampled_labels, cols=10)
            #code = torch.tensor(np.random.normal(-1, 1,
            #    (args.batch_size, args.factors))).float().cuda()
            z = utils.sample_d(x_dist, args.batch_size)
            embedding = netE(z)[1]
            
            gen_final_conv = W2(embedding, label)#, code)
            inter_acc, inter_loss = 0, 10.
            for m in gen_final_conv:
                #_, pred_label, pred_code = netD(m)
                #mi_loss, (cat_acc, cat_loss) = MI_loss(args, gt_labels, pred_label, code, pred_code)
                q_pred = netQ(m)
                pred = q_pred.max(1, keepdim=True)[1]
                inter_acc = pred.eq(gt_labels.data.view_as(pred)).long().cpu().sum()
                inter_loss = F.cross_entropy(q_pred, gt_labels)
                mi_loss = args.alpha * inter_loss
                """
                if cat_acc.item() > inter_acc:
                    inter_acc = cat_acc.item()
                if cat_loss.item() < inter_loss:
                    inter_loss = cat_loss.item()
                """
                mi_loss.backward(retain_graph=True)
            optimQ.step()
            optimE.step()
            c_acc.append(inter_acc)
            c_loss.append(inter_loss)

            if batch_idx % 50 == 0:
                acc = (correct / 1)
                print ('**************************************')
                print ('{} MNIST Test, beta: {}'.format(args.model, args.beta))
                print ('Acc: {}, Loss: {}, MI loss: {}'.format(acc, loss, mi_loss))
                print ('best test loss: {}'.format(args.best_loss))
                print ('best test acc: {}'.format(args.best_acc))
                print ('categorical acc: {}'.format(torch.tensor(c_acc, dtype=torch.float).max()/len(label)))
                print ('categorical loss: {}'.format(torch.tensor(c_loss, dtype=torch.float).max()/len(label)))
                print ('**************************************')
                c_acc, c_loss = [], []

            #if batch_idx > 1 and batch_idx % 199 == 0:
            if batch_idx % 199 == 0:
                test_acc = 0.
                test_loss = 0.
                for i, (data, target) in enumerate(mnist_test):
                    z = utils.sample_d(x_dist, args.batch_size)
                    codes = netE(z)
                    c = torch.tensor(np.random.uniform(-1, 1,
                            (args.batch_size, args.factors))).float().cuda()
                    y = to_categorical(np.random.randint(0, 10, args.batch_size),
                            cols=10).float().cuda()
                    l1 = W1(codes[0])
                    l2 = W2(codes[1], y)#, c)
                    l3 = W3(codes[2])
                    #sample1, sample2 = sample_layer(args, netE, W2, 10)
                    for (g1, g2, g3) in zip(l1, l2, l3):
                        correct, loss = train_clf(args, [g1, g2, g3], data, target)
                        test_acc += correct.item()
                        test_loss += loss.item()

                test_loss /= len(mnist_test.dataset) * args.batch_size
                test_acc /= len(mnist_test.dataset) * args.batch_size
                print ('Accuracy: {}, Loss: {}'.format(test_acc, test_loss))
                
                if test_loss < best_test_loss or test_acc > best_test_acc:
                    print ('==> new best stats, saving')
                    utils.save_clf(args, [g1, g2, g3], test_acc)
                    print ('this')
                    #args.exp = 'sample1'
                    #utils.save_clf(args, [g1, sample1, g3], test_acc)
                    #args.exp = 'sample2'
                    #utils.save_clf(args, [g1, sample2, g3], test_acc)
                    
                    #utils.save_hypernet_mnist(args, [netE, W1, W2, W3], test_acc)
                    if test_loss < best_test_loss:
                        best_test_loss = test_loss
                        args.best_loss = test_loss
                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
                        args.best_acc = test_acc


if __name__ == '__main__':

    args = load_args()
    modeldef = netdef.nets()[args.target]
    pprint.pprint (modeldef)

    # log some of the netstat quantities so we don't subscript everywhere
    args.stat = modeldef
    args.shapes = modeldef['shapes']
    train(args)

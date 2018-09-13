import sys
import torch
import pprint
import argparse
import numpy as np

from torch import optim
from torch.nn import functional as F

import ops
import utils
import netdef
import datagen

def load_args():

    parser = argparse.ArgumentParser(description='param-wgan')
    parser.add_argument('--z', default=100, type=int, help='latent space width')
    parser.add_argument('--ze', default=300, type=int, help='encoder dimension')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=200000, type=int)
    parser.add_argument('--target', default='small2', type=str)
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--beta', default=1000, type=int)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--use_x', default=False, type=bool)
    parser.add_argument('--load_e', default=False, type=bool)
    parser.add_argument('--pretrain_e', default=False, type=bool)
    parser.add_argument('--scratch', default=False, type=bool)
    parser.add_argument('--exp', default='0', type=str)
    parser.add_argument('--use_d', default=False, type=str)
    parser.add_argument('--model', default='small', type=str)
    parser.add_argument('--disc_iters', default=5, type=int)

    args = parser.parse_args()
    return args

# hard code the two layer net
def train_clf(args, Z, data, target, val=False):
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
    correct = None
    if val:
        pred = out.data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).long().cpu().sum()
    return (correct, loss)

""" conditional distribution Q(x|c) """
class Q(nn.Module):
    def __init__(self):
        super(Q, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 10),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 4*4*64)
        x = self.fc(x)
        return x


def MI_loss(args, code, y):
    z = torch.zeros(args.batch_size, args.z).cuda()
    for i in range(args.batch_size):
        z[i][y[i]*10:(y[i]*10)+9] += 1
    xentropy = F.binary_cross_entropy_with_logits(code, z)
    entropy = torch.mean(-torch.sum(z * torch.log(z + 1e-8), dim=1))
    mi_loss = xentropy + entropy
    return mi_loss


def train(args):
    
    torch.manual_seed(8734)
    netE = models.Encoder(args).cuda()
    W1 = models.GeneratorW1(args).cuda()
    W2 = models.GeneratorW2(args).cuda()
    W3 = models.GeneratorW3(args).cuda()
    netD = models.DiscriminatorZ(args).cuda()
    print (netE, W1, W2, W3)

    print (dir(models))
    optimE = optim.Adam(netE.parameters(), lr=.0005, betas=(0.5, 0.9), weight_decay=1e-4)
    optimW1 = optim.Adam(W1.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimW2 = optim.Adam(W2.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimW3 = optim.Adam(W3.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimD = optim.Adam(netD.parameters(), lr=1e-5, betas=(0.5, 0.9), weight_decay=1e-4)
    
    best_test_acc, best_test_loss = 0., np.inf
    args.best_loss, args.best_acc = best_test_loss, best_test_acc

    mnist_train, mnist_test = datagen.load_mnist(args)
    x_dist = utils.create_d(args.ze)
    z_dist = utils.create_d(args.z)
    one = torch.FloatTensor([1]).cuda()
    mone = (one * -1).cuda()
    print ("==> pretraining encoder")
    j = 0
    final = 100.
    e_batch_size = 1000
    if args.load_e:
        netE, optimE, _ = utils.load_model(args, netE, optimE)
        print ('==> loading pretrained encoder')
    if args.pretrain_e:
        for j in range(1000):
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

    print ('==> Begin Training')
    for _ in range(args.epochs):
        for batch_idx, (data, target) in enumerate(mnist_train):
            ops.batch_zero_grad([netE, W1, W2, W3, netD])
            """ generate encoding """
            z = utils.sample_d(x_dist, args.batch_size)
            codes = netE(z)
            l1 = W1(codes[0])
            l2 = W2(codes[1])
            l3 = W3(codes[2])

            """ train discriminator """
            if args.use_d:
                for _ in range(args.disc_iters):
                    d_losses = []
                    noise = utils.sample_d(z_dist, args.batch_size)
                    codes_d = netE(z)
                    for code in codes_d:
                        d_real = netD(noise)
                        d_fake = netD(code)
                        d_losses.append(-(torch.mean(d_real) - torch.mean(d_fake)))
                    for d_loss in d_losses:
                        d_loss.backward(retain_graph=True)
                    optimD.step()        
                    netD.zero_grad();
                    optimD.zero_grad()
            
            """ train generator """
            d_fake, clf_loss = [], []
            if args.use_d:
                for code in codes:
                    d_fake.append(netD(code))
            for i, (g1, g2, g3) in enumerate(zip(l1, l2, l3)):
                correct, loss = train_clf(args, [g1, g2, g3], data, target, val=True)
                clf_loss.append(loss)
                if args.use_d:
                    scaled_loss = args.beta * (loss + d_fake[i])
                else:
                    scaled_loss = args.beta * loss
                scaled_loss.backward(retain_graph=True)
            optimE.step()
            optimW1.step()
            optimW2.step()
            optimW3.step()
            loss = torch.stack(clf_loss).mean().item()

            """ MI loss """
            # want to maximize the mutaul information between the labels and a given filter
            # last conv layer only
            mi_loss = MI_loss(args, codes[1], target)
            mi_loss.backward()
            optimE.step()
            optimW3.step()

            if batch_idx % 50 == 0:
                acc = (correct / 1) 
                print ('**************************************')
                print ('{} MNIST Test, beta: {}'.format(args.model, args.beta))
                print ('Acc: {}, Loss: {}, MI loss: {}'.format(acc, loss, mi_loss))
                print ('best test loss: {}'.format(args.best_loss))
                print ('best test acc: {}'.format(args.best_acc))
                print ('**************************************')

            if batch_idx > 1 and batch_idx % 199 == 0:
                test_acc = 0.
                test_loss = 0.
                for i, (data, y) in enumerate(mnist_test):
                    z = utils.sample_d(x_dist, args.batch_size)
                    codes = netE(z)
                    l1 = W1(codes[0])
                    l2 = W2(codes[1])
                    l3 = W3(codes[2])
                    for (g1, g2, g3) in zip(l1, l2, l3):
                        correct, loss = train_clf(args, [g1, g2, g3], data, y, val=True)
                        test_acc += correct.item()
                        test_loss += loss.item()
                test_loss /= len(mnist_test.dataset) * args.batch_size
                test_acc /= len(mnist_test.dataset) * args.batch_size
                print ('Test Accuracy: {}, Test Loss: {}'.format(test_acc, test_loss))
                if test_loss < best_test_loss or test_acc > best_test_acc:
                    print ('==> new best stats, saving')
                    #utils.save_clf(args, z_test, test_acc)
                    utils.save_hypernet_mnist(args, [netE, W1, W2, W3], test_acc)
                    if test_loss < best_test_loss:
                        best_test_loss = test_loss
                        args.best_loss = test_loss
                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
                        args.best_acc = test_acc


if __name__ == '__main__':

    args = load_args()
    if args.model is 'small':
        import models.models_mnist_small as models
    #elif args.model is 'nobn':
    #    import models.models_mnist_nobn as models
    #elif args.model is 'full':
    #    import models.models_mnist as models
    else:
        raise NotImplementedError
    print (models.Encoder(args))
    modeldef = netdef.nets()[args.target]
    pprint.pprint (modeldef)
    # log some of the netstat quantities so we don't subscript everywhere
    args.stat = modeldef
    args.shapes = modeldef['shapes']
    train(args)

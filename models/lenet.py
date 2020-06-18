import torch
import torch.nn as nn
import torch.nn.functional as F
from .hypergan_base import HyperGAN_Base
import itertools

""" LeNet5 Pytorch definition """
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (5,5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5,5))
        self.linear1 = nn.Linear(16*5*5, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class LeNet_Dropout(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (5,5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5,5))
        self.linear1 = nn.Linear(16*5*5, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(F.dropout2d(self.conv1(x)), (2,2), 0.2, training=True))
        x = F.max_pool2d(F.relu(F.dropout2d(self.conv2(x)), (2,2), 0.2, training=True))
        x = x.view(-1, 16*5*5)
        x = F.relu(F.dropout(self.linear1(x), p=0.2, training=True))
        x = F.relu(F.dropout(self.linear2(x), p=0.2, training=True))
        x = self.linear3(x)
        return x



class Mixer(nn.Module):
    def __init__(self, args):
        super(Mixer, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.linear1 = nn.Linear(self.s, args.n_hidden, bias=self.bias)
        self.linear2 = nn.Linear(args.n_hidden, args.n_hidden, bias=self.bias)
        self.linear3 = nn.Linear(args.n_hidden, self.z*5, bias=self.bias)
        if args.use_bn:
            self.bn1 = nn.BatchNorm1d(args.n_hidden)
            self.bn2 = nn.BatchNorm1d(args.n_hidden)
        else:
            self.bn1 = lambda x: x
            self.bn2 = lambda x: x

    def forward(self, x):
        x = x.view(-1, self.s) #flatten filter size
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        x = x.view(-1, 5, self.z)
        w = torch.stack([x[:, i] for i in range(5)])
        return w


class GeneratorW1(nn.Module):
    def __init__(self, args):
        super(GeneratorW1, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.linear1 = nn.Linear(self.z, args.n_hidden, bias=self.bias)
        self.linear2 = nn.Linear(args.n_hidden, args.n_hidden, bias=self.bias)
        self.linear3 = nn.Linear(args.n_hidden, 150 + 6, bias=self.bias)
        if args.use_bn:
            self.bn1 = nn.BatchNorm1d(args.n_hidden)
            self.bn2 = nn.BatchNorm1d(args.n_hidden)
        else:
            self.bn1 = lambda x: x
            self.bn2 = lambda x: x

    def forward(self, x):
        if not self.bias:
            self.bn1.bias.data.zero_()
            self.bn2.bias.data.zero_()
        x = torch.zeros_like(x).normal_(0, 0.01) + x
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        w, b = x[:, :150], x[:, -6:]
        w = w.view(-1, 6, 1, 5, 5)
        b = b.view(-1, 6)
        return (w, b)


class GeneratorW2(nn.Module):
    def __init__(self, args):
        super(GeneratorW2, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.linear1 = nn.Linear(self.z, args.n_hidden, bias=self.bias)
        self.linear2 = nn.Linear(args.n_hidden, args.n_hidden, bias=self.bias)
        self.linear3 = nn.Linear(args.n_hidden, 2400+16, bias=self.bias)
        if args.use_bn:
            self.bn1 = nn.BatchNorm1d(args.n_hidden)
            self.bn2 = nn.BatchNorm1d(args.n_hidden)
        else:
            self.bn1 = lambda x: x
            self.bn2 = lambda x: x

    def forward(self, x):
        if not self.bias:
            self.bn1.bias.data.zero_()
            self.bn2.bias.data.zero_()
        x = torch.zeros_like(x).normal_(0, 0.01) + x
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        w, b = x[:, :2400], x[:, -16:]
        w = w.view(-1, 16, 6, 5, 5)
        b = b.view(-1, 16)
        return (w, b)


class GeneratorW3(nn.Module):
    def __init__(self, args):
        super(GeneratorW3, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.linear1 = nn.Linear(self.z, args.n_hidden, bias=self.bias)
        self.linear2 = nn.Linear(args.n_hidden, args.n_hidden, bias=self.bias)
        self.linear3 = nn.Linear(args.n_hidden, 120*400+120, bias=self.bias)
        if args.use_bn:
            self.bn1 = nn.BatchNorm1d(args.n_hidden)
            self.bn2 = nn.BatchNorm1d(args.n_hidden)
        else:
            self.bn1 = lambda x: x
            self.bn2 = lambda x: x

    def forward(self, x):
        if not self.bias:
            self.bn1.bias.data.zero_()
            self.bn2.bias.data.zero_()
        x = torch.zeros_like(x).normal_(0, 0.01) + x
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        w, b = x[:, :400*120], x[:, -120:]
        w = w.view(-1, 120, 400)
        b = b.view(-1, 120)
        return (w, b)


class GeneratorW4(nn.Module):
    def __init__(self, args):
        super(GeneratorW4, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.linear1 = nn.Linear(self.z, args.n_hidden, bias=self.bias)
        self.linear2 = nn.Linear(args.n_hidden, args.n_hidden, bias=self.bias)
        self.linear3 = nn.Linear(args.n_hidden, 84*120+84, bias=self.bias)
        if args.use_bn:
            self.bn1 = nn.BatchNorm1d(args.n_hidden)
            self.bn2 = nn.BatchNorm1d(args.n_hidden)
        else:
            self.bn1 = lambda x: x
            self.bn2 = lambda x: x

    def forward(self, x):
        if not self.bias:
            self.bn1.bias.data.zero_()
            self.bn2.bias.data.zero_()
        x = torch.zeros_like(x).normal_(0, 0.01) + x
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        w, b = x[:, :120*84], x[:, -84:]
        w = w.view(-1, 84, 120)
        b = b.view(-1, 84)
        return (w, b)


class GeneratorW5(nn.Module):
    def __init__(self, args):
        super(GeneratorW5, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.linear1 = nn.Linear(self.z, args.n_hidden, bias=self.bias)
        self.linear2 = nn.Linear(args.n_hidden, args.n_hidden, bias=self.bias)
        self.linear3 = nn.Linear(args.n_hidden, 10*84+10, bias=self.bias)
        if args.use_bn:
            self.bn1 = nn.BatchNorm1d(args.n_hidden)
            self.bn2 = nn.BatchNorm1d(args.n_hidden)
        else:
            self.bn1 = lambda x: x
            self.bn2 = lambda x: x

    def forward(self, x):
        if not self.bias:
            self.bn1.bias.data.zero_()
            self.bn2.bias.data.zero_()
        x = torch.zeros_like(x).normal_(0, 0.01) + x
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        w, b = x[:, :10*84], x[:, -10:]
        w = w.view(-1, 10, 84)
        b = b.view(-1, 10)
        return (w, b)


class DiscriminatorZ(nn.Module):
    def __init__(self, args):
        super(DiscriminatorZ, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.linear1 = nn.Linear(self.z, args.n_hidden)
        self.linear2 = nn.Linear(args.n_hidden, args.n_hidden)
        self.linear3 = nn.Linear(args.n_hidden, 1)

    def forward(self, x):
        x = x.view(-1, self.z)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        x = torch.sigmoid(x)
        return x


class HyperGAN(HyperGAN_Base):
    
    def __init__(self, args, device):
        super(HyperGAN, self).__init__(args)
        self.device = device
        self.mixer = Mixer(args).to(device)
        self.generator = self.Generator(args, device)
        self.discriminator = DiscriminatorZ(args).to(device)
        self.model = LeNet().to(device)

    class Generator(object):
        def __init__(self, args, device):
            self.W1 = GeneratorW1(args).to(device)
            self.W2 = GeneratorW2(args).to(device)
            self.W3 = GeneratorW3(args).to(device)
            self.W4 = GeneratorW4(args).to(device)
            self.W5 = GeneratorW5(args).to(device)

        def __call__(self, x):
            w1, b1 = self.W1(x[0])
            w2, b2 = self.W2(x[1])
            w3, b3 = self.W3(x[2])
            w4, b4 = self.W4(x[3])
            w5, b5 = self.W5(x[4])
            layers = [w1, b1, w2, b2, w3, b3, w4, b4, w5, b5]
            return layers
        
        def as_list(self):
            return [self.W1, self.W2, self.W3, self.W4, self.W5]

    def attach_optimizers(self, lr_m, lr_g, lr_d=None):
        self.optim_mixer = torch.optim.Adam(self.mixer.parameters(), lr=lr_m, weight_decay=1e-4)
        if lr_d:
            self.optim_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d, weight_decay=1e-4)
        gen_params = [g.parameters() for g in self.generator.as_list()]
        self.optim_generator = torch.optim.Adam(itertools.chain(*gen_params), lr=lr_g, weight_decay=1e-4)

    def update_generator(self):
        self.optim_generator.step()

    """ functional model for training """
    def eval_f(self, Z, data):
        w1, b1, w2, b2, w3, b3, w4, b4, w5, b5 = Z
        x = F.conv2d(data, w1, stride=1, padding=2, bias=b1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.conv2d(x, w2, stride=1, bias=b2)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 16*5*5)
        x = F.relu(F.linear(x, w3, bias=b3))
        x = F.relu(F.linear(x, w4, bias=b4))
        x = F.linear(x, w5, bias=b5)
        return x

    def restore_models(self, args):
        d = torch.load(args.resume)
        self.mixer.load_state_dict(d['mixer']['state_dict'])
        self.discriminator.load_state_dict(d['Dz']['state_dict'])
        generators = self.generator.as_list()
        for i, gen in enumerate(generators):
            gen.load_state_dict(d['W{}'.format(i)]['state_dict'])


    def save_models(self, args, metrics=None):
        save_dict = {
                'mixer': {'state_dict': self.mixer.state_dict()},
                'W1': {'state_dict': self.generator.W1.state_dict()},
                'W2': {'state_dict': self.generator.W2.state_dict()},
                'W3': {'state_dict': self.generator.W3.state_dict()},
                'W4': {'state_dict': self.generator.W4.state_dict()},
                'W5': {'state_dict': self.generator.W5.state_dict()},
                'netD': {'state_dict': self.discriminator.state_dict()}
                }
        path = 'saved_models/mnist/lenet-{}-{}.pt'.format(args.exp, metrics)
        torch.save(save_dict, path)

    def print_hypergan(self):
        print (self.mixer)
        for generator in self.generator.as_list():
            print (generator)

    def zero_grad(self):
        self.mixer.zero_grad()
        self.discriminator.zero_grad()
        for generator in self.generator.as_list():
            generator.zero_grad()

    def train_(self):
        self.mixer.train()
        self.discriminator.train()
        for generator in self.generator.as_list():
            generator.train()

    def eval_(self):
        self.mixer.eval()
        self.discriminator.eval()
        for generator in self.generator.as_list():
            generator.eval()

from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'Encoder_agent'
        self.linear1 = nn.Linear(self.ze, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, 6*self.z)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(1024)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        # print ('E in: ', x.shape)
        x = x.view(-1, self.ze) #flatten filter size
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        x = x.view(-1, 6, self.z)
        w1 = x[:, 0]
        w2 = x[:, 1]
        w3 = x[:, 2]
        w4 = x[:, 3]
        w5 = x[:, 4]
        w6 = x[:, 5]
        # print ('E out: ', x.shape)
        return w1, w2, w3, w4, w5, w6


class DiscriminatorZ(nn.Module):
    def __init__(self, args):
        super(DiscriminatorZ, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        
        self.name = 'Discriminator_z'
        self.linear1 = nn.Linear(self.z, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 1024)
        self.linear4 = nn.Linear(1024, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print ('Dz in: ', x.shape)
        x = x.view(self.batch_size, -1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        x = self.sigmoid(x)
        # print ('Dz out: ', x.shape)
        return x


""" Convolutional (1 x 32 x 3 x 3) """
class GeneratorW1(nn.Module):
    def __init__(self, args):
        super(GeneratorW1, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW1'
        self.linear1 = nn.Linear(self.z, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 288)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        #print ('W1 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        #x = self.linear4(x)
        x = x.view(-1, 32, 1, 3, 3)
        #print ('W1 out: ', x.shape)
        return x


""" Convolutional (32 x 32 x 3 x 3) """
class GeneratorW2(nn.Module):
    def __init__(self, args):
        super(GeneratorW2, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW2'
        self.linear1 = nn.Linear(self.z, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.linear3 = nn.Linear(2048, 4096)
        self.linear4 = nn.Linear(4096, 9216)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(2048)
        self.bn3 = nn.BatchNorm1d(4096)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        #print ('W3 in : ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.relu(self.bn3(self.linear3(x)))
        x = self.linear4(x)
        x = x.view(-1, 32, 32, 3, 3)
        #print ('W3 out: ', x.shape)
        return x


""" Convolutional (32 x 32 x 3 x 3) """
class GeneratorW3(nn.Module):
    def __init__(self, args):
        super(GeneratorW3, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW3'
        self.linear1 = nn.Linear(self.z, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.linear3 = nn.Linear(2048, 4096)
        self.linear4 = nn.Linear(4096, 9216)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(2048)
        self.bn3 = nn.BatchNorm1d(4096)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        #print ('W3 in : ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.relu(self.bn3(self.linear3(x)))
        x = self.linear4(x)
        x = x.view(-1, 32, 32, 3, 3)
        #print ('W3 out: ', x.shape)
        return x


""" Convolutional (32 x 32 x 3 x 3) """
class GeneratorW4(nn.Module):
    def __init__(self, args):
        super(GeneratorW4, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW4'
        self.linear1 = nn.Linear(self.z, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.linear3 = nn.Linear(2048, 4096)
        self.linear4 = nn.Linear(4096, 9216)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(2048)
        self.bn3 = nn.BatchNorm1d(4096)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        #print ('W3 in : ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.relu(self.bn3(self.linear3(x)))
        x = self.linear4(x)
        x = x.view(-1, 32, 32, 3, 3)
        #print ('W3 out: ', x.shape)
        return x


""" Linear (800 x 1) """
class GeneratorW5(nn.Module):
    def __init__(self, args):
        super(GeneratorW5, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW5'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 800)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        #print ('W4 in : ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        x = x.view(-1, 1, 800)
        #print ('W4 out: ', x.shape)
        return x

""" Linear (800 x n_actions) """
class GeneratorW6(nn.Module):
    def __init__(self, args):
        super(GeneratorW6, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorW6'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, 2048)
        self.linear4 = nn.Linear(2048, 800*self.n_actions)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(2048)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        #print ('W4 in : ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.relu(self.bn3(self.linear3(x)))
        x = self.linear4(x)
        x = x.view(-1, self.n_actions, 800)
        #print ('W4 out: ', x.shape)
        return x

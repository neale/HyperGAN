import sys
import numpy as np
import torch
import utils
from train_mnist import train, test, Net, WideNet, WideNet7
from glob import glob
import natsort
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


class NetSlice(nn.Module):
    def __init__(self):
        super(NetSlice, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 128, 7),
                nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(128, 256, 7),
                nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                nn.MaxPool2d(2, 2),
                )
        self.fc1 = nn.Sequential(
                nn.Linear(2*2*256, 1024), 
                nn.ReLU(True),
                )
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        # return x
        x = self.conv2[0](x)
        return x


def gallery(array, ncols=3):
    nindex, height, width = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols))
    return result


kwargs = {'num_workers': 1, 'pin_memory': True}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=32, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=32, shuffle=True, **kwargs)

id = np.random.randint(200)
model = WideNet7().cuda()
model2 = NetSlice().cuda()
paths = natsort.natsorted(glob('./models/mnist/wide7/*.pt'))
model.load_state_dict(torch.load(paths[id]))
print ("****** original accuracy: ")
test(model, 0)

""" TEST RANDOM """
np.random.seed(1)
# params = np.random.normal(size=(128, 1, 3, 3))
# params = np.random.normal(size=(256, 128, 3, 3))
# params = np.random.normal(size=(32, 1, 3, 3))
# params = np.random.normal(size=(64, 32, 3, 3))
# params = np.random.normal(size=(256, 128, 7, 7))
# params = np.random.normal(size=(128, 1, 7, 7))
""" TEST GAN """
# params = np.load('./params/sampled/mnist/wide/conv1/params_iter_11500.npy')
# params = np.load('./params/sampled/mnist/wide/conv2/params_iter_4100.npy')
# params = np.load('./params/sampled/mnist/wide7/conv2/params_iter_4100.npy')
params = np.load('./params/sampled/mnist/wide7/conv2/params_iter_700.npy')
# utils.plot_histogram(params, save=True)
print (params.shape)
from scipy.misc import imshow, imsave
state = model.state_dict()
conv2 = state['conv2.0.weight']
state['conv2.0.weight'] = torch.Tensor(params).cuda()
model.load_state_dict(state)

# test(model, 0)

""" fix weights and fine tune """
model.conv2[0].weight.requires_grad=False
model.conv2[0].bias.requires_grad=False
model.conv1[0].weight.requires_grad=False
model.conv1[0].bias.requires_grad=False

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
criterion = nn.CrossEntropyLoss()
print ("****** initial test ****** ")
# test(model, 0)
for epoch in range(10000):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).cuda(), Variable(target).cuda()
        optimizer.zero_grad()
        output = model(data)

       
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    test(model, epoch)
    """
    if epoch % 5 == 0:
        state = model.state_dict()
        model2.load_state_dict(state)
        output2 = model2(data)
        print (output2.shape)
        output2 = output2.data.cpu().numpy().reshape((-1, 5, 5))[:121]
        res = gallery(output2, ncols=11)
        #imshow(res)
        imsave('./weights/generated_mnist_wide7_conv2/{}.png'.format(epoch), res)
    """



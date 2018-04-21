import numpy as np
import torch
import utils
from train_cifar import train, test, Net, WideNet
from glob import glob
import natsort
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from vgg16 import VGG

kwargs = {'num_workers': 1, 'pin_memory': True}

transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.CIFAR10(root='./data', train=True,
        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=32,
        shuffle=True, num_workers=1)

testset = datasets.CIFAR10(root='./data', train=False,
        download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=32,
        shuffle=False, num_workers=1)


id = np.random.randint(2)
model = Net().cuda()
#model = VGG().cuda()
paths = natsort.natsorted(glob('./models/cifar/1x/*.pt'))
model.load_state_dict(torch.load(paths[id]))
print (test(model, 0))

""" TEST RANDOM """
np.random.seed(1)
# params = np.random.normal(size=(128, 64, 3, 3))
# params = np.random.normal(size=(3, 64, 3, 3))
# params = np.random.normal(size=(64, 64, 3, 3))
""" TEST WGAN """
params = np.load('./params/sampled/cifar/conv2/params_iter_21900.npy')

state = model.state_dict()
conv2 = state['conv2.weight']
state['conv2.weight'] = torch.Tensor(params).cuda()
# print (state['conv2.weight'] - conv2)
model.load_state_dict(state)

""" fix weights and fine tune """
model.conv2.weight.requires_grad=False
model.conv2.bias.requires_grad=False
model.conv1.weight.requires_grad=False
model.conv1.bias.requires_grad=False
    
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
criterion = nn.CrossEntropyLoss()
print ("****** initial test ****** ")
acc = test(model, 0)
print ("Accuracy: {}".format(acc))
for epoch in range(10000):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).cuda(), Variable(target).cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    acc = test(model, epoch)
    print ("Training epoch: {} -- accuracy: {}".format(epoch, acc))

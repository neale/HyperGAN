import sys
import operator
import numpy as np
from glob import glob

import torch
import utils
import natsort
import torchvision
import torch.nn as nn
import torch.optim as optim
from scipy.misc import imshow, imsave
from torch.autograd import Variable
from torchvision import datasets, transforms
from presnet import PreActResNet18, ResNetSlice18
from train_cifar import train, test, Net, WideNet
import presnet

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
        download=True,
        transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
        shuffle=True,
        num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
        download=True,
        transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
        shuffle=False, num_workers=2)


kwargs = {'num_workers': 1, 'pin_memory': True}


def load_model(rand=False, id=None):
    if id is None:
        id = np.random.randint(40)
    model = presnet.ResNetVis18().cuda()
    paths = natsort.natsorted(glob('./models/cifar/presnet/*.pt'))
    path = './models/cifar/presnet/cifar_0.pt'
    model.load_state_dict(torch.load(path))
    print (test(model))

    if rand is True:
        np.random.seed(3)
        params = np.random.normal(size=(512, 512, 3, 3))
        # params = np.random.normal(size=(128, 64, 3, 3))
        # params = np.random.normal(size=(3, 64, 3, 3))
        # params = np.random.normal(size=(64, 64, 3, 3))
    else:
        params = np.load('./params/sampled/cifar/presnet/layer4.1.conv2/params_iter_61800.npy')

    state = model.state_dict()
    # conv2 = state['conv7.weight']
    # state['conv2.weight'] = torch.Tensor(params).cuda()
    # print (state['conv2.weight'] - conv2)
    conv2 = state['features.4.1.conv2.weight']
    state['features.4.1.conv2.weight'] = torch.Tensor(params).cuda()
    model.load_state_dict(state)

    """ Fix Gradients """
    model.conv1.weight.requires_grad=False
    try:
        model.conv1.bias.requires_grad=False
    except:
        pass
    conv_names = [x for x in list(state.keys()) if 'conv' in x]
    for name in conv_names:
        terms = name.split('.')
        name = '.'.join(terms[:-1])
        operator.attrgetter(name)(model).weight.requires_grad = False
        if operator.attrgetter(name)(model).bias is not None:
            battr.bias.requires_grad = False
       
    acc = test(model)
    print ("****** initial test ****** ")
    print ("Accuracy: {}".format(acc))   


def gallery(array, ncols=3):
    nindex, height, width = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width)
            .swapaxes(1,2)
            .reshape(height*nrows, width*ncols))
    return result


def insert_params(model, type):
    np.random.seed(3)
    if type == 'random':
        params = np.random.normal(size=(512, 512, 3, 3))
    if type == 'sampled':
        params = np.load('params/sampled/cifar/presnet/layer4.1.conv2/params_iter_1500.npy')
    state = model.state_dict()
    conv = state['layer4.1.conv2.weight']
    state['layer4.1.conv2.weight'] = torch.Tensor(params).cuda()
    model.load_state_dict(state)
    return model


# Following functions only make sense for resnet model
def view_activations():
    model = ResNetSlice18().cuda()
    model_id = np.random.randint(40)
    paths = natsort.natsorted(glob('./models/cifar/presnet/*.pt'))
    model.load_state_dict(torch.load(paths[model_id]))
    model = insert_params(model, 'sampled')
    for i, (data, _) in enumerate(trainloader):
        data = Variable(data).cuda()
        outputs = model(data)
        print ("Activations shape: {}".format(outputs.size()))
        acts = outputs.view(-1, 4, 4)[:16]
        acts = acts.data.cpu().numpy()
        imgs = gallery(acts, ncols=4)
        imshow(imgs)
        sys.exit(0)

        


def view_weights(model):
    pass


for n in np.random.randint(0, 40, size=(10,)):
    load_model(rand=True, id=n)
# view_activations()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=0.0000001, momentum=0.9,
                      weight_decay=5e-4)

criterion = nn.CrossEntropyLoss()

for epoch in range(10000):
    model.train()
    acc = 0
    total = 0
    correct = 0
    train_loss = 0
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = Variable(data).cuda(), Variable(target).cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        total += target.size(0)
        _, predicted = torch.max(output.data, 1)
        correct += predicted.eq(target.data).cpu().sum()
        acc = 100. * correct / total
        utils.progress_bar(batch_idx, len(trainloader), 
                           'Loss: {} | Acc: {}'.format(train_loss/(batch_idx+1), acc))
    acc = test(model)
    print ("Test Accuracy: ", acc)

import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils, transforms
import torchvision
import cv2
import sys
import numpy as np
import argparse
import presnet
from scipy.misc import imshow

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            print (name, self.target_layers)
            x = module(x)
            if name in self.target_layers:
                print (name)
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.features, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model.linear(output)
        return target_activations, output


def preprocess_image(img):

    preprocessed_img = img.copy()[: , :, ::-1]
    print (preprocessed_img.shape)
    preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (0, 1, 2)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad = True)
    return input


def show_cam_on_image(img, mask):
    print (mask)
    heatmap = cv2.applyColorMap(np.uint8(255.*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255.
    cam = heatmap + np.float32(img).transpose(1, 2, 0)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input) 

    def __call__(self, input, index = None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.features.zero_grad()
        self.model.linear.zero_grad()
        one_hot.backward(retain_variables=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis = (2, 3))[0, :]
        cam = np.ones(target.shape[1 : ], dtype = np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (32, 32))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input), torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1), positive_mask_2)

        return grad_input

class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model.features._modules[idx] = GuidedBackpropReLU()

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index = None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

            if index == None:
                index = np.argmax(output.cpu().data.numpy())

            one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
            one_hot[0][index] = 1
            one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
            if self.cuda:
                one_hot = torch.sum(one_hot.cuda() * output)
            else:
                one_hot = torch.sum(one_hot * output)

            # self.model.features.zero_grad()
            # self.model.classifier.zero_grad()
            one_hot.backward(retain_variables=True)

            output = input.grad.cpu().data.numpy()
            output = output[0,:,:,:]

            return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
            help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
            help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

if __name__ == '__main__':
    args = get_args()

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

    for i, (img, target) in enumerate(trainloader):
        img = img[0]
        grad_cam = GradCam(model = presnet.ResNetVis18(), target_layer_names = ['4'], use_cuda=args.use_cuda)
        img = img.numpy()
        
        imshow(img)
        target_index = None
        input = preprocess_image(img)
        print (input)
        mask = grad_cam(input, target_index)
        print (mask)

        show_cam_on_image(img, mask)

        gb_model = GuidedBackpropReLUModel(model=presnet.ResNetVis18(), use_cuda=args.use_cuda)
        gb = gb_model(input, index=target_index)
        utils.save_image(torch.from_numpy(gb), 'gb.jpg')

        cam_mask = np.zeros(gb.shape)
        for i in range(0, gb.shape[0]):
            cam_mask[i, :, :] = mask

        cam_gb = np.multiply(cam_mask, gb)
        utils.save_image(torch.from_numpy(cam_gb), 'cam_gb.jpg')

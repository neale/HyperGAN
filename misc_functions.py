"""
Created on Thu Oct 21 11:09:09 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import copy
import cv2
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision import models
import presnet

def convert_to_grayscale(cv2im):
    """
        Converts 3d image to grayscale

    Args:
        cv2im (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(cv2im), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def save_gradient_images(gradient, file_name):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    if not os.path.exists('results'):
        os.makedirs('results')
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    gradient = np.uint8(gradient*255).transpose(1, 2, 0)
    path_to_file = os.path.join('results', file_name + '.jpg')
    # Convert RBG to GBR
    gradient = gradient[..., ::-1]
    print (gradient.shape)
    cv2.imwrite(path_to_file, gradient)


def save_class_activation_on_image(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original image

    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists('results'):
        os.makedirs('results')
    # Grayscale activation map
    path_to_file = os.path.join('results', file_name+'_Cam_Grayscale.jpg')
    cv2.imwrite(path_to_file, activation_map)
    # Heatmap of activation map
    activation_heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_HSV)
    path_to_file = os.path.join('results', file_name+'_Cam_Heatmap.jpg')
    cv2.imwrite(path_to_file, activation_heatmap)
    # Heatmap on picture
    org_img = cv2.resize(org_img, (32, 32))
    img_with_heatmap = np.float32(activation_heatmap) + np.float32(org_img)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    path_to_file = os.path.join('results', file_name+'_Cam_On_Image.jpg')
    cv2.imwrite(path_to_file, np.uint8(255 * img_with_heatmap))


def preprocess_image(cv2im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    # Resize image
    if resize_im:
        cv2im = cv2.resize(cv2im, (32, 32))
    im_as_arr = np.float32(cv2im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing

    Args:
        im_as_var (torch variable): Image to recreate

    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.4914, -0.4822, -0.4465]
    reverse_std = [1/0.2023, 1/0.1994, 1/0.2010]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    # Convert RBG to GBR
    recreated_im = recreated_im[..., ::-1]
    return recreated_im


def get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize

    returns:
        pos_saliency ( )
    """
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency


def insert_params(model, type):
    np.random.seed(3)
    if type == 'random':
        params = np.random.normal(size=(512, 512, 3, 3))
    if type == 'sampled':
        params = np.load('params/sampled/cifar/presnet/layer4.1.conv2/params_iter_61800.npy')
    state = model.state_dict()
    conv = state['features.4.1.conv2.weight']
    state['features.4.1.conv2.weight'] = torch.Tensor(params).cuda()
    model.load_state_dict(state)
    return model


def get_params(example_index):
    # Pick one of the examples
    print('==> Preparing data..')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
            download=True,
            transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
            shuffle=False, num_workers=2)


    example_list = []
    for (inputs, labels) in testloader:
        img = inputs[example_index]
        target = labels[example_index]
        break

    target_class = target
    file_name_to_export = ['']
    # Read image
    from scipy.misc import imshow, imsave
    imshow(img.numpy())
    imsave('./results/orig.png', img.numpy().transpose(1, 2, 0))
    original_image = img
    prep_img = original_image.unsqueeze_(0)
    # Convert to Pytorch variable
    prep_img = Variable(prep_img, requires_grad=True)
    # Process image
    pretrained_model = presnet.ResNetVis18()
    pretrained_model.load_state_dict(torch.load('./models/cifar/presnet/cifar_0.pt'))
    pretrained_model = insert_params(pretrained_model, type='sampled')
    print (prep_img.size())
    img = original_image.numpy()[0].transpose(1, 2, 0)
    cv2.imwrite('./results/original.png', img)
    return (original_image,
            prep_img,
            target_class,
            file_name_to_export,
            pretrained_model)

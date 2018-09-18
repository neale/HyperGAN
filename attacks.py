import os
import foolbox
import numpy as np
import scipy.misc
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
# from tools.cifar_base import cifar_model
from foolbox.criteria import TargetClass, Misclassification
from foolbox.criteria import TopKMisclassification


def display_im(img, title=None):

    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.show()


def load_model(model):
    # MNIST
    mean = np.array([.1307]).reshape((1, 1, 1))
    std = np.array([.3081]).reshape((1, 1, 1))
    fmodel = foolbox.models.PyTorchModel(model, 
            bounds=(0, 1), num_classes=10, preprocessing=(mean, std))
    return fmodel


def build_attack(attack_str, model, criterion):
    if attack_str == 'deepfool':
        return foolbox.attacks.DeepFoolAttack(model)
    if attack_str == 'lbfgs':
        return foolbox.attacks.LBFGSAttack(model, criterion)
    if attack_str == 'fgsm':
        return foolbox.attacks.FGSM(model, criterion)
    if attack_str == 'mim':
        return foolbox.attacks.MIM(model, criterion)
    else:
        print ("model not supported")
        sys.exit(0)


def generate(model, image, targets):
    label = np.argmax(model.predictions(image))
    if targets == 2 and label == 0:
        print ("classified real as adversarial")
        return None
    target_class = get_target(label, targets)
    criterion = Misclassification()

    attack = build_attack('lbfgs', model, criterion)
    return attack

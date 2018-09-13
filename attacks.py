import os
import foolbox
import keras
import numpy as np
import scipy.misc
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
# from tools.cifar_base import cifar_model
from foolbox.criteria import TargetClass, Misclassification
from foolbox.criteria import TopKMisclassification
from keras.backend.tensorflow_backend import set_session


def config_tf(gpu=0.8):
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu
    set_session(tf.Session(config=config))
    keras.backend.set_learning_phase(0)


def display_im(img, title=None):

    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.show()


def load_model(model, type=None):
    if type is 'resnet':  # resnet
        preprocessing = (np.array([104, 116, 123]), 1)
        fmodel = foolbox.models.KerasModel(model, bounds=(0, 255), preprocessing=preprocessing)
        return fmodel
    if type is 'inception':
        preprocessing = (0, np.array([1., 1., 1.]))
        fmodel = foolbox.models.KerasModel(model, bounds=(-1, 1), preprocessing=preprocessing)
        return fmodel
    # kmodel = cifar_model(top=top, path=weights)
    # kmodel.summary()
    # fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 255))
    # return fmodel


def get_target(label, r):

    if r == 2:
        target = 0
        return target

    target = label
    while target == label:
        target = np.random.randint(r)
    return target


def print_stats(image, original, model):

    failed = 0
    l1 = np.argmax(model.predictions(image))
    l2 = np.argmax(model.predictions(original))
    print "Adversarial Predicted: ", l1
    print "Original Predicted: ", l2
    if l1 == l2:
        failed = 1
    return failed


def score_dataset(model, x, label):

    misses = 0
    for img in x:
        pred = np.argmax(model.predictions(img))
        if pred != label:
            misses += 1
            print pred, label
    return misses


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
        print "model not supported"
        sys.exit(0)


def generate(model, image, targets):

    label = np.argmax(model.predictions(image))
    # print "label: ", label
    if targets == 2 and label == 0:
        print "classified real as adversarial"
        return None
    target_class = get_target(label, targets)
    #print "Target Class: {}".format(target_class)
    # criterion = TargetClass(target_class)
    criterion = Misclassification()
    # criterion = TopKMisclassification(k=3)

    attack = build_attack('lbfgs', model, criterion)
    return attack

    # attack = foolbox.attacks.GradientAttack(model, criterion)
    try:
        adversarial = attack(image, label, unpack=False, epsilons=[0.001])
    except ValueError:
        return None
    return adversarial
    # return adversarial
    # except:
    #     print "FAILED"
    #     return None

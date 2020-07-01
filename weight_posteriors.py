import numpy as np
import torch
import matplotlib.pyplot as plt
from time import sleep
import seaborn as sns

""" 
we sample a collection of weights and graph their distribution
weights can be sampled over individual neurons, layers, or everything
"""
def sample_weight_posteriors_mednet(args, hypergan, epoch):
    # collect 100 samples
    n = 50
    samples = 1
    posterior = torch.zeros(n, samples*90506)
    posterior_qz = torch.zeros(n, samples*args.z*args.ngen)
    l1_z_posterior = torch.zeros(n, samples*args.z)
    l2_z_posterior = torch.zeros(n, samples*args.z)
    l3_z_posterior = torch.zeros(n, samples*args.z)
    l4_z_posterior = torch.zeros(n, samples*args.z)
    l5_z_posterior = torch.zeros(n, samples*args.z)
    l1_posterior = torch.zeros(n, samples*32*3*3*3)
    l2_posterior = torch.zeros(n, samples*32*64*3*3)
    l3_posterior = torch.zeros(n, samples*64*64*3*3)
    l4_posterior = torch.zeros(n, samples*256*128)
    l5_posterior = torch.zeros(n, samples*128*10)
    sample_neurons = [torch.zeros(n, samples) for _ in range(8)]
    indexes = np.random.randint(1000, size=(8,))
    for i in range(n):
        s = torch.randn(args.batch_size, args.s).to(args.device)
        z = torch.randn(args.batch_size, args.z).to(args.device)
        codes = hypergan.mixer(s)
        params = hypergan.generator(codes)
        all_weights = torch.cat([x.contiguous()[:samples].view(-1) for x in params]).view(-1)
        posterior[i] = all_weights
        l1_posterior[i] = params[0].contiguous()[:samples].view(-1)
        l2_posterior[i] = params[2].contiguous()[:samples].view(-1)
        l3_posterior[i] = params[4].contiguous()[:samples].view(-1)
        l4_posterior[i] = params[6].contiguous()[:samples].view(-1)
        l5_posterior[i] = params[8].contiguous()[:samples].view(-1)
        
        l1_z_posterior[i] = codes[0][:samples].view(-1)
        l2_z_posterior[i] = codes[1][:samples].view(-1)
        l3_z_posterior[i] = codes[2][:samples].view(-1)
        l4_z_posterior[i] = codes[3][:samples].view(-1)
        l5_z_posterior[i] = codes[4][:samples].view(-1)
        posterior_qz[i] = codes[:, :samples, :].contiguous().view(-1)
        for j in range(8):
            sample_neurons[j][i] = params[2].contiguous()[:samples].view(-1)[indexes[j]]

    fig, ax = plt.subplots(4, 5, figsize=(30, 30))
    # plot layer outputs
    for i, item in enumerate([l1_posterior, l2_posterior, l3_posterior, l4_posterior, l5_posterior]):
        sns.distplot(item.cpu().detach().view(-1), ax=ax[0, i])
        ax[0, i].set_yticks([])
        ax[0, i].set_title('Layer {}'.format(i))
    # plot mixer outputs per layer
    for i, item in enumerate([l1_z_posterior, l2_z_posterior, l3_z_posterior, l4_z_posterior, l5_z_posterior]):
        sns.distplot(item.cpu().detach().view(-1), ax=ax[1, i])
        ax[1, i].set_yticks([])
        ax[1, i].set_title('Q(s)-{}'.format(i))
    # plot all weights
    sns.distplot(posterior.cpu().detach().view(-1), ax=ax[2,0])
    ax[2, 0].set_yticks([])
    ax[2, 0].set_title('Full G(z)')
    # plot aggregated Q(s)
    sns.distplot(posterior_qz.cpu().detach().view(-1), ax=ax[2,1])
    ax[2, 1].set_yticks([])
    ax[2, 1].set_title('Full Q(s)')
    # plot individual neurons
    for i, item in enumerate(sample_neurons[:3]):
        sns.distplot(item.cpu().detach().view(-1), ax=ax[2,i+2])
        ax[2, i+2].set_yticks([])
        ax[2, i+2].set_title('Neuron-{}'.format(indexes[i]))
    for i, item in enumerate(sample_neurons[3:]):
        sns.distplot(item.cpu().detach().view(-1), ax=ax[3,i])
        ax[3, i].set_yticks([])
        ax[3, i].set_title('Neuron-{}'.format(indexes[i]+3))

    plt.savefig('mednet-weight-posterior-epoch-{}'.format(epoch))
    plt.close('all')
    #plt.show()

def sample_weight_posteriors_small(args, hypergan, epoch):
    # collect 100 samples
    n = 50
    samples = 1
    posterior = torch.zeros(n, samples*31594)
    posterior_qz = torch.zeros(n, samples*args.z*args.ngen)
    l1_z_posterior = torch.zeros(n, samples*args.z)
    l2_z_posterior = torch.zeros(n, samples*args.z)
    l3_z_posterior = torch.zeros(n, samples*args.z)
    l1_posterior = torch.zeros(n, samples*32*1*5*5)
    l2_posterior = torch.zeros(n, samples*32*32*5*5)
    l3_posterior = torch.zeros(n, samples*512*10)
    sample1 = torch.zeros(n, samples)
    sample2 = torch.zeros(n, samples)
    sample3 = torch.zeros(n, samples)
    sample4 = torch.zeros(n, samples)
    for i in range(n):
        s = torch.randn(args.batch_size, args.s).to(args.device)
        z = torch.randn(args.batch_size, args.z).to(args.device)
        codes = hypergan.mixer(s)
        params = hypergan.generator(codes)
        all_weights = torch.cat([x.contiguous()[:samples].view(-1) for x in params]).view(-1)
        posterior[i] = all_weights

        l1_posterior[i] = params[0].contiguous()[:samples].view(-1)
        l2_posterior[i] = params[2].contiguous()[:samples].view(-1)
        l3_posterior[i] = params[4].contiguous()[:samples].view(-1)

        l1_z_posterior[i] = codes[0][:samples].view(-1)
        l2_z_posterior[i] = codes[1][:samples].view(-1)
        l3_z_posterior[i] = codes[2][:samples].view(-1)
        posterior_qz[i] = codes[:, :samples, :].contiguous().view(-1)

        sample1[i] = params[2].contiguous()[:samples].view(-1)[10]
        sample2[i] = params[2].contiguous()[:samples].view(-1)[50]
        sample3[i] = params[2].contiguous()[:samples].view(-1)[100]
        sample4[i] = params[2].contiguous()[:samples].view(-1)[300]

    fig, ax = plt.subplots(4, 3, figsize=(30, 30))
    # plot layer outputs
    for i, item in enumerate([l1_posterior, l2_posterior, l3_posterior]):
        sns.distplot(item.cpu().detach().view(-1), ax=ax[0, i])
        ax[0, i].set_yticks([])
        ax[0, i].set_title('Layer {}'.format(i))
    # plot mixer outputs per layer
    for i, item in enumerate([l1_z_posterior, l2_z_posterior, l3_z_posterior]):
        sns.distplot(item.cpu().detach().view(-1), ax=ax[1, i])
        ax[1, i].set_yticks([])
        ax[1, i].set_title('Q(s)-{}'.format(i))
    # plot all weights
    sns.distplot(posterior.cpu().detach().view(-1), ax=ax[2,0])
    ax[2, 0].set_yticks([])
    ax[2, 0].set_title('Full G(z)')
    # plot aggregated Q(s)
    sns.distplot(posterior_qz.cpu().detach().view(-1), ax=ax[2,1])
    ax[2, 1].set_yticks([])
    ax[2, 1].set_title('Full Q(s)')
    # plot individual neurons
    sns.distplot(sample1.cpu().detach().view(-1), ax=ax[2,2])
    ax[2, 2].set_yticks([])
    ax[2, 2].set_title('Neuron 1')
    for i, item in enumerate([sample2, sample3, sample4]):
        sns.distplot(item.cpu().detach().view(-1), ax=ax[3,i])
        ax[3, i].set_yticks([])
        ax[3, i].set_title('Neuron-{}'.format(i+1))

    plt.savefig('small-weight-posterior-epoch-{}'.format(epoch))
    plt.close('all')
    #plt.show()


def sample_weight_posteriors_lenet(args, hypergan, epoch):
    # collect 100 samples
    n = 25
    posterior = torch.zeros(100, 31594)
    posterior_qz = torch.zeros(100, args.z*args.ngen)
    l1_z_posterior = torch.zeros(100, args.z)
    l2_z_posterior = torch.zeros(100, args.z)
    l3_z_posterior = torch.zeros(100, args.z)
    l4_z_posterior = torch.zeros(100, args.z)
    l5_z_posterior = torch.zeros(100, args.z)
    l1_posterior = torch.zeros(100, 32*1*5*5)
    l2_posterior = torch.zeros(100, 32*32*5*5)
    l3_posterior = torch.zeros(100, 512*10)
    l4_posterior = torch.zeros(100, 512*10)
    l5_posterior = torch.zeros(100, 512*10)
    for i in range(n):
        s = torch.randn(args.batch_size, args.s).to(args.device)
        z = torch.randn(args.batch_size, args.z).to(args.device)
        codes = hypergan.mixer(s)
        params = hypergan.generator(codes)
        all_weights = torch.stack(params).view(-1)

        l1_posterior[i] = params[0].view(-1)
        l2_posterior[i] = params[2].view(-1)
        l3_posterior[i] = params[4].view(-1)
        l4_posterior[i] = params[6].view(-1)
        l5_posterior[i] = params[8].view(-1)
        
        l1_z_posterior[i] = codes[0]
        l2_z_posterior[i] = codes[1]
        l3_z_posterior[i] = codes[2]
        l4_z_posterior[i] = codes[3]
        l5_z_posterior[i] = codes[4]

    fig, ax = plt.subplots(3, 5, figsize=(40, 40))
    # plot layer outputs
    for i, item in enumerate([l1_posterior, l2_posterior, l2_posterior, l4_posterior, l5_posterior]):
        sns.distplot(item.view(-1), ax=ax[0, i])
        ax[0, i].set_yticks([])
        ax[0, i].set_title('Layer {}'.format(i))
    # plot mixer outputs per layer
    for i, item in enumerate([l1_z_posterior, l2_z_posterior, l3_z_posterior, l4_z_posterior, l5_z_posterior]):
        sns.distplot(item.view(-1), ax=ax[1, i])
        ax[1, i].set_yticks([])
        ax[1, i].set_title('Q(s)-{}'.format(i))
    # plot all weights
    sns.distplot(posterior.view(-1), ax=[2,0])
    ax[2, 0].set_yticks([])
    ax[2, 0].set_title('Full G(z)')
    # plot aggregated Q(s)
    sns.distplot(posterior_qz.view(-1), ax=[2,1])
    ax[2, 1].set_yticks([])
    ax[2, 1].set_title('Full Q(s)')
    plt.tight_layout()
    plt.show()


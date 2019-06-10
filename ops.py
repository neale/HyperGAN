import math
import torch
import torch.nn.functional as F
import torch.autograd as autograd


def pretrain_encoder(args, mixer, optimM):
    for enc_iter in range(1000):
        s = torch.randn(1000, args.s).to(args.device)
        codes = mixer(s).view(args.z*args.ngen, 1000).transpose(0, 1)
        full_z = torch.randn(1000, args.z*args.ngen).to(args.device)
        mean_loss, cov_loss = pretrain_loss(codes, full_z)
        loss = mean_loss + cov_loss
        loss.backward()
        optimM.step()
        mixer.zero_grad()
        print ('Pretrain Enc iter: {}, Mean Loss: {}, Cov Loss: {}'.format(
            enc_iter, mean_loss.item(), cov_loss.item()))
        if loss.item() < 0.1:
            print ('Finished Pretraining Encoder')
            break
    return


def pretrain_loss(encoded, noise):
    mean_z = torch.mean(noise, dim=0, keepdim=True)
    mean_e = torch.mean(encoded, dim=0, keepdim=True)
    mean_loss = F.mse_loss(mean_z, mean_e)
    cov_z = torch.matmul((noise-mean_z).transpose(0, 1), noise-mean_z)
    cov_z /= 999
    cov_e = torch.matmul((encoded-mean_e).transpose(0, 1), encoded-mean_e)
    cov_e /= 999
    cov_loss = F.mse_loss(cov_z, cov_e)
    return mean_loss, cov_loss



def log_density(z, z_var):
    z_dim = z.size(1)
    z = -(z_dim/2)*math.log(2*math.pi*z_var) + z.pow(2).sum(1).div(-2*z_var)
    return z


def calc_d_loss(args, Dz, z, codes, cifar=False):
    log_pz = log_density(z, 2).view(-1, 1).to(args.device)
    dim = args.batch_size * args.ngen
    zeros = torch.zeros(dim, 1, requires_grad=True).to(args.device)
    ones = torch.ones(args.batch_size, 1, requires_grad=True).to(args.device)
    d_z = Dz(z)
    codes = codes.view(-1, args.z)
    d_codes = Dz(codes)
    log_pz_ = log_density(torch.ones(dim, 1), 2).view(-1, 1).to(args.device)
    d_loss = F.binary_cross_entropy_with_logits(d_z+log_pz, ones) + \
             F.binary_cross_entropy_with_logits(d_codes+log_pz_, zeros)
    total_loss = d_loss
    return total_loss, d_codes

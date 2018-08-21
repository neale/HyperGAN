import args
import models
import hypera2c as H
import utils_xai as utils
from atari_data import MultiEnvironment

import warnings
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch, os, gym, time, glob, argparse, sys

from torch.optim import Adam
from scipy.misc import imresize
from scipy.signal import lfilter

os.environ['OMP_NUM_THREADS'] = '1'


def discount(rewards, gamma):
    rewards = rewards[:, ::-1]
    result = lfilter([1], [1, -gamma], rewards)
    return result[:, ::-1]


def printlog(args, s, end='\n', mode='a'):
    print(s, end=end) 
    f=open(args.save_dir+'log.txt',mode)
    f.write(s+'\n')
    f.close()

map_gpu = {
        'cuda:0': 'cuda:0',
        'cuda:1': 'cuda:0',
        'cuda:2': 'cuda:0',
        'cuda:3': 'cuda:0',
        'cuda:4': 'cuda:0',
        'cuda:5': 'cuda:0',
        'cuda:6': 'cuda:0',
        'cuda:7': 'cuda:0',
        'cpu': 'cpu',
}


class HyperNetwork(object):
    def __init__(self, args):
        #super(HyperNetwork, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'HyperNetwork'
        self.encoder = models.Encoder(args).cuda()
        self.adversary = models.DiscriminatorZ(args).cuda()
        self.generators = [
                models.GeneratorW1(args).cuda(),
                models.GeneratorW2(args).cuda(),
                models.GeneratorW3(args).cuda(),
                models.GeneratorW4(args).cuda(),
                models.GeneratorW5(args).cuda(),
                models.GeneratorW6(args).cuda()
        ]
        
    def set_test_mode(self):
        self.encoder.eval()
        self.adversary.eval()
        for gen in self.generators:
            gen.eval()

    def set_train_mode(self):
        self.encoder.train()
        self.adversary.train()
        for gen in self.generators:
            gen.train()

        # sync weights with another hypernet
    def sync(self, H2):
        self.encoder.load_state_dict(H2.encoder.state_dict())
        self.adversary.load_state_dict(H2.adversary.state_dict())
        for g1, g2 in zip(self.generators, H2.generators):
            g1.load_state_dict(g2.state_dict())

    def save_state(self, optim, num_frames, mean_reward):
        path = 'models/HyperGAN/atari/{}/agent.pt'.format(self.env)
        Hypernet_dict = {
            'E': utils.get_net_dict(self.encoder, optim['optimE']),
            'D': utils.get_net_dict(self.adversary, optim['optimD']),
            'W1': utils.get_net_dict(self.generators[0], optim['optimG'][0]),
            'W2': utils.get_net_dict(self.generators[1], optim['optimG'][1]),
            'W3': utils.get_net_dict(self.generators[2], optim['optimG'][2]),
            'W4': utils.get_net_dict(self.generators[3], optim['optimG'][3]),
            'W5': utils.get_net_dict(self.generators[4], optim['optimG'][4]),
            'W6': utils.get_net_dict(self.generators[5], optim['optimG'][5]),
            'num_frames': num_frames,
            'mean_reward': mean_reward
        }
        torch.save(Hypernet_dict, path)

    def load_state(self, optim):
        layer_names = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6']
        modules = [self.generators[0], self.generators[1], self.generators[2],
                self.generators[3], self.generators[4], self.generators[5]]

        optimizers = [optim['optimG'][0], optim['optimG'][1], optim['optimG'][2],
                optim['optimG'][3], optim['optimG'][4], optim['optimG'][5]]
        path = 'models/HyperGAN/atari/{}/agent.pt'.format(self.env)
        HN = torch.load(path)
        objectE = utils.open_net_dict(HN['E'], self.encoder, optim['optimE'])
        self.encoder, optim['optimE'] = objectE
        objectD = utils.open_net_dict(HN['D'], self.adversary, optim['optimD'])
        self.adversary, optim['optimD'] = objectD

        for (l, net, op) in zip(layer_names, modules, optimizers):
            old_state = net.state_dict()
            net, op = utils.open_net_dict(HN[l], net, op)
            assert net.state_dict != old_state

        num_frames = HN['num_frames']
        mean_reward = HN['mean_reward']
        return optim, num_frames, mean_reward


def FuncPolicy(args, W, state):
    x = F.relu(F.conv2d(state, W[0], stride=2, padding=1))
    x = F.relu(F.conv2d(x, W[1], stride=2, padding=1))
    x = F.relu(F.conv2d(x, W[2], stride=2, padding=1))
    x = F.relu(F.conv2d(x, W[3], stride=2, padding=1))
    x = x.view(x.size(0), -1)
    return  F.linear(x, W[4]), F.linear(x, W[5])


def cost_func(args, values, logps, actions, rewards):
    np_values = values.cpu().data.numpy()
    delta_t = np.asarray(rewards) + args.gamma * np_values[:,1:] - np_values[:,:-1]
    gae = discount(delta_t, args.gamma * args.tau)
    logpys = logps.gather(2, torch.tensor(actions).view(actions.shape[0],-1,1))
    policy_loss = -(logpys.view(gae.shape[0],-1) * torch.tensor(gae.copy(),
        dtype=torch.float32).cuda()).sum()
    
    # l2 loss over value estimator
    rewards[:,-1] += args.gamma * np_values[:,-1]
    discounted_r = discount(np.asarray(rewards), args.gamma)
    discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32).cuda()
    value_loss = .5 * (discounted_r - values[:,:-1]).pow(2).sum()

    entropy_loss = -(-logps * torch.exp(logps)).sum().cuda() # encourage lower entropy
    # print ('policy: ', policy_loss, 'value loss: ', value_loss, 'ent loss: ', entropy_loss)
    loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
    return loss


def load_optim(args, HyperNet):
    gen_optim = []
    w = 1e-4
    if args.test: 
        lr_e, lr_d, lr_g = 0, 0, 0
    else:
        lr_e, lr_d, lr_g = 5e-3, 5e-4, 1e-5
    for p in HyperNet.generators:
        gen_optim.append(Adam(p.parameters(), lr=lr_g, betas=(.9,.999), weight_decay=w))

    Optim = { 
        'optimE': Adam(HyperNet.encoder.parameters(), lr=lr_e, betas=(.5,.999),
            weight_decay=w, eps=1e-8),
        'optimD': Adam(HyperNet.adversary.parameters(), lr=lr_d, betas=(.9,.999),
            weight_decay=w),
        'optimG': gen_optim,
    }
    return Optim


def pretrain_e(args, HyperNet, Optim):
    HyperNet.encoder, Optim = H.pretrain_encoder(args, HyperNet.encoder, Optim)
    return HyperNet, Optim


def train(args, envs, model, optim):
    info = {k: torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 
        'run_loss', 'episodes', 'frames']}
    if args.resume:
        res = model.load_state(optim)
        print ('loaded agent')
        if res is not None:
            optim, num_frames, mean_reward = res
            info['frames'] += num_frames * 1e6
    else:
        if args.pretrain_e:
            pretrain_e(args, model, optim)
    if int(info['frames'].item()) == 0:
        printlog(args,'', end='', mode='w')

    print ('=> loaded HyperGAN networks')
    Fmodel = FuncPolicy # a local/unshared model
    hypernet = HyperNetwork(args)
    state_shape = (args.batch_size, 1, 80, 80)
    state = torch.tensor(envs.reset()).view(state_shape).cuda() # get first state
    start_time = last_disp_time = time.time()
    episode_length = np.zeros(args.batch_size)
    epr, eploss = np.zeros(args.batch_size), np.zeros(args.batch_size)
    values, logps, actions, rewards = [], [], [], []
    print ('=> starting training')
    i = 0
    if args.test:
        hypernet.set_test_mode()
        envs.set_monitor()
        envs.envs[0].reset()

    while info['frames'][0] <= 8e7 or args.test: 
        i += 1
        episode_length += 1
        # get network weights
        weights, hypernet, optim = H.get_policy_weights(args, hypernet, optim)
        # compute the agent response with generated weights
        
        value, logit = Fmodel(args, weights, state)
        logp = F.log_softmax(logit, dim=-1)
        # print ('=> updating state')
        action = torch.exp(logp).multinomial(num_samples=1).data
        state, reward, done, _ = envs.step(action)
        if args.render:
            envs.envs[0].render()

        state = torch.tensor(state).view(state_shape).cuda()
        reward = np.clip(reward, -1, 1)
        epr += reward
        done = done or episode_length >= 1e4 # don't playing one ep for too long
        info['frames'] += args.batch_size
        num_frames = int(info['frames'].item())
        if num_frames % 1e6 == 0: # save every 2M frames
            printlog(args, '\n\t{:.0f}M frames: saved model\n'.format(num_frames/1e6))
            model.save_state(optim, num_frames/1e6, info['run_epr'].item())
        done_count = np.sum(done)
        if done_count > 0:
            if done[0] == True and time.time() - last_disp_time > 5:
                timenow = time.gmtime(time.time() - start_time)
                elapsed = time.strftime("%Hh %Mm %Ss", timenow)
                printlog(args,'frames {:.1f}M, mean epr {:.2f}, run loss {:.2f}'
                    .format(num_frames/1e6, info['run_epr'].item(), info['run_loss'].item()))
                print (action.view(action.numel()))
                last_disp_time = time.time()

            for j, d in enumerate(done):
                if d:
                    info['episodes'] += 1
                    interp = 1 if info['episodes'][0] == 1 else 1 - args.horizon
                    info['run_loss'].mul_(1-interp).add_(interp*eploss[j])
                    info['run_epr'].mul_(1-interp).add_(interp * epr[j])
                    episode_length[j], epr[j], eploss[j] = 0, 0, 0

        values.append(value)
        logps.append(logp)
        actions.append(action)
        rewards.append(reward)

        if i % args.rnn_steps == 0:
            weights, hypernet, optim = H.get_policy_weights(args, hypernet, optim)
            next_value = Fmodel(args, weights, state)[0]
            if done_count > 0:
                for item, ep in enumerate(done):
                    if ep:
                        next_value[item] = 0
            values.append(next_value.detach().cuda())
            values = torch.cat(values, dim=1)
            actions = torch.cat(actions, dim=1)
            logps = torch.stack(logps, dim=1)
            rewards = np.transpose(np.asarray(rewards))
            loss = cost_func(args, values, logps, actions, rewards)
            # print ('A2C Loss: ', loss)
            eploss += loss.item()
            # print ('=> Updating Optimizer')
            model, optim = H.update_hn(args, loss, model, optim)
            values, logps, actions, rewards = [], [], [], []
            

if __name__ == "__main__":
    
    args = args.load_args()
    args.save_dir = '{}/'.format(args.env.lower()) 
    if args.render:  
        args.processes = 1 
        args.test = True 
    if args.test:  
        args.lr = 0

    args.n_actions = gym.make(args.env).action_space.n
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir) 

    # print ('=> Multienvironment settings')
    envs = MultiEnvironment(args.env, args.batch_size, args.frame_skip)
    torch.manual_seed(args.seed)
    torch.cuda.device(args.gpu)

    model = HyperNetwork(args)
    optim = load_optim(args, model)
    train(args, envs, model, optim)

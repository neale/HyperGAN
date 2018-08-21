# Generate some frames of Pong
# Generate some frames of Pong

import random
import sys
import numpy as np
import gym
import torch
from torch.autograd import Variable
from copy import deepcopy
from scipy.misc import imresize
from multiprocessing import Pool
from concurrent import futures

prepro = lambda img: imresize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255.

def map_fn(fn, *iterables):
    with futures.ThreadPoolExecutor(max_workers=8) as executor:
        result_iterator = executor.map(fn, *iterables)
    return [i for i in result_iterator]


class MultiEnvironment():
    def __init__(self, name, batch_size, fskip = 0):
        self.batch_size = batch_size
        self.envs = [] #map(lambda idx: gym.make(name), range(batch_size))
        self.name = name
        for i in range(batch_size):
            env = gym.make(name)
            env.seed(i)
            if fskip > 0: env.unwrapped.frameskip = fskip
            self.envs.append(env)

    def reset(self, index=None):
        if index is not None:
            return self._reset_i(index)
        return np.array([prepro(env.reset()) for env in self.envs])

    def _reset_i(self, i):
        return prepro(self.envs[i].reset())

    def get_action_size(self, env_name = None):
        return self.envs[0].action_space.n

    def only_one_env(self):
        self.envs = [self.envs[0]]
    
    def set_monitor(self):
        # only one env here
        from gym import wrappers
        self.envs[0] = wrappers.Monitor(self.envs[0], 'tmp/{}/'.format(self.name), force=True)

    def step(self, actions):
        assert len(actions) == len(self.envs)

        def run_one_step(env, action):
            state, reward, done, info = env.step(action)
            if done:
                state = env.reset()
            return prepro(state), reward, done, info

        results = map_fn(run_one_step, self.envs, actions)
        states, rewards, dones, infos = zip(*results)
        return np.array(states), rewards, dones, infos


if __name__ == '__main__':
    batch_size = 64
    env = MultiEnvironment('Pong-v0', batch_size)
    for i in range(10):
        actions = np.random.randint(0, 4, size=batch_size)
        states, rewards, dones, infos = env.step(actions)

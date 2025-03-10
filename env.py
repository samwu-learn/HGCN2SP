import os
import pickle
import numpy as np
import torch
import random
import torch.multiprocessing as mp
from utils import solve_cflp_softmax_new

class CFLPEnv:
    def __init__(self, data, bs, clusters, k, batch_size, process=2, device=None):
        self.data = data
        self.used = list(range(len(data)))
        self.bs = bs
        self.clusters = clusters
        self.k = k
        self.batch_size = batch_size
        self.process = process
        self.device = device

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.used)
        self.loc = 0
        self.batch_data = []
        self.batch_bs = []
        self.batch_clusters = []
        for start in range(0, len(self.data), self.batch_size):
            end = start + self.batch_size
            if end >= len(self.data):
                end = len(self.data)
            self.batch_data.append([self.data[i] for i in self.used[start:end]])
            self.batch_bs.append([self.bs[i] for i in self.used[start:end]])
            self.batch_clusters.append([self.clusters[i] for i in self.used[start:end]])
        return self.batch_data[self.loc], {}

    def rein_reward(self, action, iteration, alpha=0.001):
        action = torch.from_numpy(action)
        chunks = action.unbind(0)
        param_values = self.batch_clusters[self.loc]
        with mp.Pool(processes=self.process) as pool:
            results = pool.map(solve_cflp_softmax_new, zip(param_values, chunks, [True]*self.batch_size))
        rewards = torch.stack(results)
        base_cost = torch.from_numpy(np.array(self.batch_bs[self.loc]))
        cost = torch.sum((base_cost[:, :-1] == rewards[:, :-2]), dim=1)/(rewards.shape[-1]-2)
        time = rewards[:, -2]
        rewards = 25 * (cost - alpha * time)
        return rewards.numpy()

    def step(self, action, iteration):
        reward = self.rein_reward(action, iteration)
        terminations = np.array([True]*self.batch_size)
        truncations = np.array([True]*self.batch_size)
        info = {}
        self.loc += 1
        if self.loc >= len(self.batch_data):
            obs, _ = self.reset()
        else:
            obs = self.batch_data[self.loc]
        return obs, reward, terminations, truncations, info

    def close(self):
        print("EndEnv...")
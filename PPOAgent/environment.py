import gym
from gym import spaces
from bpp_utils import *
from bpp_algorithms import best_fit, gen_random_items
from problem_generator import get_problems
import os
from datetime import datetime


class BPPEnv(gym.Env):
    def __init__(self, **kwargs):
        super().__init__()
        self.test = kwargs.get('test')
        self.data = kwargs.get('data')

        # instance generation args
        self.gen_mode = kwargs.get('gen_mode')
        self.gen_n = kwargs.get('gen_n')
        self.gen_std = kwargs.get('gen_std')
        self.gen_probs = kwargs.get('gen_probs')

    def step(self, action):
        reward = 0
        # translate action into bin index
        valid_indexes = np.where(self.action_mask == True)[0]
        action = valid_indexes[action]

        assert np.sum(self.bins[action]) + self.current_item <= 1.0 + 1e-6
        self.bins[action].append(self.current_item)
        if int(np.sum(self.bins[action])) == 1:
            reward += 0.001

        self.packed_item_count += 1
        terminate = self._next_item()

        # termination
        if not terminate:
            self._update_action_mask()
        else:
            # terminal reward
            reward = self.bf_result - bins_num(self.bins)
            # reward = np.exp(utilization(self.bins))
        return self.state(), reward, terminate, {}

    def reset(self):
        if self.test:
            self.items = self.data.pop(0)
        else:
            self.items = get_problems(mode=self.gen_mode, n_items=self.gen_n,
                                      std=self.gen_std, probs=self.gen_probs)

        self.current_item = 0
        self.bf_result = best_fit([], self.items)
        self.bins = []
        self.histogram = [0] * 11
        self.action_mask = []

        self._next_item()
        self._update_action_mask()
        self.packed_item_count = 0
        return self.state()

    def state(self):
        available_bins = [b for i, b in enumerate(self.bins) if self.action_mask[i]]
        bin_spec = np.zeros(shape=(len(available_bins), 8))

        # bin_spec[0, :] = np.array(self.histogram) / (self.packed_item_count + 1)
        # current bins_num
        for i in range(len(available_bins)):
            b = available_bins[i]
            idx = i
            # capacity left
            bin_spec[idx, 0] = 1 - np.sum(b)
            # item num
            bin_spec[idx, 1] = len(b) / self.packed_item_count if self.packed_item_count > 0 else 0
            # item mean
            bin_spec[idx, 2] = np.average(b) if len(b) > 0 else 0
            # item std
            bin_spec[idx, 3] = np.std(b) if len(b) > 0 else 0
            # min item
            bin_spec[idx, 4] = np.min(b) if len(b) > 0 else 0
            # max item
            bin_spec[idx, 5] = np.max(b) if len(b) > 0 else 0
            # item range
            bin_spec[idx, 6] = bin_spec[idx, 5] - bin_spec[idx, 4]
            # utilization if pack here
            bin_spec[idx, 7] = np.sum(b) + self.current_item

        # print('bins_num:')
        # print(bin_spec)
        return bin_spec

    def _update_action_mask(self):
        if len(self.bins) == 0 or len(self.bins[-1]) > 0:
            self.bins.append([])
        mask = np.array([False] * len(self.bins))
        for i in range(len(self.bins)):
            if np.sum(self.bins[i]) + self.current_item <= 1.0 + 1e-6:
                mask[i] = True

        self.action_mask = mask

    def _next_item(self):
        if len(self.items) > 0:
            self.current_item = self.items[0]
            self.items = self.items[1:]
            self.histogram[int(self.current_item // 5 - 2)] += 1
            return False
        else:
            self.current_item = 0
            return True

    def is_terminal(self):
        return len(self.items) == 0 and self.current_item == 0

    def get_action_length_mask(self):
        return np.array([m for m in self.action_mask if m])

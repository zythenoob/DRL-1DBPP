from copy import deepcopy

import numpy as np

from bpp_utils import *
import c_bpp as cb

max_bin_opening = 20
history_size = 0
state_size = (3 + history_size, max_bin_opening, 50)

epsilon = 1e-6


class State:
    def __init__(self, items=None, copy=False):
        if items is None and not copy:
            raise ValueError('Must input the problem in initialization')
        self.bins = [[] for _ in range(max_bin_opening)]
        self.bf_bins = [[] for _ in range(max_bin_opening)]
        self.items = items
        self.history_items = []
        self.history_actions = []
        self.current_item = 0
        self.packed_item_count = 0
        self.p = 0
        self.reward = 0
        self.sampler_actions = 50
        self.r = 'bf'

        if copy:
            self.bf_result = 0
            self.mask = []
        else:
            self.bf_result = cb.best_fit([], self.items)
            self.mask = self.get_action_mask()

    def actions(self):
        if self.p == 0:
            # adversary move
            return hist2items([1] * self.sampler_actions)
        elif self.p == 1:
            # player move
            return np.array([i for i in range(max_bin_opening) if self.mask[i]])

    # def state(self):
    #     state_features = np.zeros(shape=state_size)
    #     end_pos = []
    #     # current bins
    #     for i, b in enumerate(self.bins):
    #         cur = 0
    #         for j in b:
    #             length = int(round(j * state_size[2]))
    #             length = 1 if length == 0 else length
    #             end = cur + length
    #             end = state_size[2] if end > state_size[2] else end
    #             state_features[0, i, cur:end] = j
    #             cur = end
    #         end_pos.append(cur)
    #
    #     # current item
    #     length = int(round(self.current_item * state_size[2]))
    #     length = 1 if length == 0 else length
    #     for i, m in enumerate(self.mask):
    #         if m:
    #             if end_pos[i] == state_size[2]:
    #                 end_pos[i] -= length
    #             end = end_pos[i] + length
    #             end = state_size[2] if end > state_size[2] else end
    #             state_features[1, i, end_pos[i]:end] = self.current_item
    #
    #     # bf bins
    #     for i, b in enumerate(self.bf_bins):
    #         cur = 0
    #         for j in b:
    #             length = int(round(j * state_size[2]))
    #             length = 1 if length == 0 else length
    #             end = cur + length
    #             end = state_size[2] if end > state_size[2] else end
    #             state_features[2, i, cur:end] = j
    #             cur = end
    #
    #     # # history items
    #     # for i, h in enumerate(self.history_actions):
    #     #     start = int(round(h[0] * state_size[2]))
    #     #     end = int(round((h[0] + h[1]) * state_size[2]))
    #     #     state_features[2+i, h[2], start:end] = h[1]
    #     #
    #     # for i in state_features:
    #     #     plt.imshow(i)
    #     #     plt.show()
    #
    #     return state_features

    def state(self):
        bin_spec = np.zeros(shape=(max_bin_opening, 8))

        # current bins_num
        for i in range(len(self.bins)):
            b = self.bins[i]
            # capacity left
            bin_spec[i, 0] = 1 - np.sum(b)
            # item num
            bin_spec[i, 1] = len(b) / self.packed_item_count if self.packed_item_count > 0 else 0
            # item mean
            bin_spec[i, 2] = np.average(b) if len(b) > 0 else 0
            # item std
            bin_spec[i, 3] = np.std(b) if len(b) > 0 else 0
            # min item
            bin_spec[i, 4] = np.min(b) if len(b) > 0 else 0
            # max item
            bin_spec[i, 5] = np.max(b) if len(b) > 0 else 0
            # item range
            bin_spec[i, 6] = bin_spec[i, 5] - bin_spec[i, 4]
            # utilization if pack here
            if self.mask[i]:
                # utilization if pack here
                bin_spec[i, 7] = np.sum(b) + self.current_item
            else:
                # utilization if pack here
                bin_spec[i, 7] = -1

        # print('bins_num:')
        # print(bin_spec)
        # bin_spec = bin_spec[np.newaxis, :, :]
        return bin_spec

    def child_state(self, action):
        new_state = deepcopy(self)
        new_state.step(action)
        return new_state

    def step(self, action):
        """
            player 0: problem instance
            player 1: agent
        """
        if self.p == 0:
            if len(self.items) > 0:
                self.current_item = action
                self.items.pop(0)
            self.mask = self.get_action_mask()
            self.p = 1
        elif self.p == 1:
            if np.sum(self.bins[action]) + self.current_item > 1.0 + epsilon:
                print(self)
                print(action)
                raise ValueError('Exceeding bin capacity')

            # self.history_actions.append((np.sum(self.bins[action]), self.current_item, action))
            # if len(self.history_actions) > history_size:
            #     self.history_actions.pop(0)

            # bf_action = cb.bf_step(bins=self.bf_bins, item=self.current_item)
            # self.bf_bins[bf_action].append(self.current_item)

            self.packed_item_count += 1

            self.bins[action].append(self.current_item)
            self.history_items.append(self.current_item)
            self.current_item = 0
            self.p = 0
        else:
            raise ValueError('Player must be either 0 or 1')

        if self.terminal():
            # self.reward = utilization(self.bins)
            if self.r == 'bf':
                bf_result = cb.best_fit([], self.history_items)
                if bins_num(self.bins) > bf_result:
                    self.reward = -1
                elif bins_num(self.bins) < bf_result:
                    self.reward = 1
                else:
                    self.reward = 0
            else:
                self.reward = utilization(self.bins)

    def get_action_mask(self):
        mask = np.array([False] * max_bin_opening)
        for i, b in enumerate(self.bins):
            if len(b) == 0:
                mask[i] = True
                break
            mask[i] = True if np.sum(b) + self.current_item <= 1.0 + epsilon else False
        return mask

    def get_available_bins(self):
        return self.bins[self.mask == True]

    def terminal(self):
        return self.p == 0 and len(self.items) == 0

    def set_sampler(self, n):
        self.sampler_actions = n

    def set_reward(self, r):
        self.r = r

    def __str__(self):
        s = ["bins: %s" % self.bins, "current item: %s" % self.current_item, "items: %s" % self.items,
             "possible actions: %s" % self.actions()]
        return "%s: \n%s" % (self.__class__.__name__, '\n'.join(s))

    def plot(self):
        image = self.state()
        for i in image:
            plt.imshow(i)
            plt.show()

    def __deepcopy__(self, memodict={}):
        copy_state = State(copy=True)
        copy_state.bins = [b.copy() for b in self.bins]
        copy_state.bf_bins = [b.copy() for b in self.bf_bins]
        copy_state.items = self.items.copy()
        copy_state.history_items = self.history_items.copy()
        copy_state.history_actions = self.history_actions.copy()
        copy_state.current_item = self.current_item
        copy_state.packed_item_count = self.packed_item_count
        copy_state.p = self.p
        copy_state.r = self.r
        copy_state.reward = self.reward
        copy_state.bf_result = self.bf_result
        copy_state.mask = self.mask.copy()
        return copy_state



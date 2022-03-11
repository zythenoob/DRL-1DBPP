import random
import torch
from copy import deepcopy

from bpp_utils import *
import c_bpp as cb


def random_policy(agent, node):
    rollout_state = deepcopy(node.state)
    while not rollout_state.terminal():
        try:
            action = random.choice(rollout_state.actions())
            # print('rollout action', action)
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(node.state))
        rollout_state.step(action=action)
    return rollout_state.reward


def bf_predict_policy(agent, node):
    rollout_state = deepcopy(node.state)
    while not rollout_state.terminal():
        if rollout_state.p == 0:
            try:
                action = random.choice(rollout_state.actions())
                # print('rollout action', action)
            except IndexError:
                raise Exception("Non-terminal state has no possible actions: " + str(node.state))
        else:
            action = cb.bf_step(rollout_state.bins, rollout_state.current_item)
        rollout_state.step(action=action)
    return rollout_state.reward


def approximate_value_policy(agent, node):
    if node.state.terminal():
        return node.state.reward
    if not node.evaluated:
        agent.get_node_value(node=node)
    # rollout_v = bf_predict_policy(agent, node)
    return node.v

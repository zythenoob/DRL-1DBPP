import numpy as np

from MCTSAgent.MCTS import MCTS
from MCTSAgent.State import *
from MCTSAgent.ReplayMemory import ReplayMemory, Transition
from MCTSAgent.Policy import random_policy, approximate_value_policy, bf_predict_policy
from MCTSAgent.main import search_device
from c_bpp import bf_step
import torch
from tqdm import tqdm


class Agent:
    def __init__(self, model, exploration=2, batch_size=32, search_iteration=1000, sampler_actions=50, rollout=bf_predict_policy, t_reward='bf'):
        self.model = model
        self.memory = ReplayMemory(capacity=5000)
        self.exploration = exploration
        self.batch_size = batch_size
        self.search_iterations = search_iteration
        self.sampler_actions = sampler_actions
        self.t_reward = t_reward
        self.rollout = rollout
        self.uct = False

    def observe(self, state):
        if len(state.actions()) > 1 and len(state.items) > 0:
            # mcts search
            state.set_sampler(self.sampler_actions)
            state.set_reward(self.t_reward)
            mcts = MCTS(self, exploration=self.exploration, rollout_policy=self.rollout)
            mcts.set_root(state)
            for m in range(self.search_iterations):
                # torch.cuda.empty_cache()
                mcts.execute()

            # print('')
            # print('single step search time:')
            # print('all:', mcts.search_time)
            # print('select:', mcts.select_time / mcts.search_time)
            # print('simulate:', mcts.simulate_time / mcts.search_time)
            # print('bp:', mcts.backpropagate_time / mcts.search_time)

            # TODO: increase mcts efficiency
            action, probs = mcts.search()
            value = mcts.root.v
            return action, value, probs
        elif len(state.actions()) == 1 or len(state.items) == 0:
            return None, None, None
        else:
            raise ValueError('Invalid state')

    def act(self, state, action):
        if action is None:
            action = random.choice(state.actions())
        # make move
        state = state.child_state(action=action)
        return state

    def experience(self, writer=None):
        exp = self.memory.sample(batch_size=self.batch_size)
        batch = Transition(*zip(*exp))
        s = np.array(batch.s, dtype=np.float32)
        m = np.array(batch.m, dtype=np.bool_)
        v = np.array(batch.v, dtype=np.float32)
        p = np.array(batch.p, dtype=np.float32)
        self.model.update(s, m, v, p, writer)

    def evaluate_node(self, node, exploration):
        scores = {}
        if self.uct:
            for action, child in node.children.items():
                scores[action] = child.q_value() + np.sqrt(np.log(node.num_visit)/child.num_visit)
        else:
            if not node.evaluated:
                self.get_node_value(node=node)

            probs = node.child_p

            if node.parent is None and exploration != 0:
                # if is root node, add Dirichlet noise for extra exploration
                noise = 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
                for action, child in node.children.items():
                    scores[action] = child.q_value() + exploration * (0.75 * probs[action] + noise[action]) * \
                                     np.sqrt(node.num_visit) / (1 + child.num_visit)
            else:
                for action, child in node.children.items():
                    scores[action] = child.q_value() + exploration * probs[action] * \
                                     np.sqrt(node.num_visit) / (1 + child.num_visit)

        return scores

    def get_node_value(self, node):
        with torch.no_grad():
            p = self.model.predict(node.state.state(), node.state.mask, device=search_device)
            # node.v = v.cpu().detach().numpy()[0][0]
            node.child_p = p.cpu().detach().numpy()[0]
            node.evaluated = True

    def set_search(self, iteration):
        self.search_iterations = iteration


class BestFitAgent(Agent):
    def __init__(self):
        super().__init__(model=None)

    def observe(self, state):
        return bf_step(state.bins, state.current_item), None, None

    def evaluate_node(self, node, exploration):
        bf_action = bf_step(node.state.bins, node.state.current_item)
        scores = {}
        for action, child in node.children.items():
            if action == bf_action:
                scores[action] = 1
            else:
                scores[action] = 0
        return scores

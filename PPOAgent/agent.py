import time
import os
from copy import deepcopy
import random

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm, trange

from PPOAgent.environment import BPPEnv
from PPOAgent.model import PointerNet, masked_softmax, pad_tensor
from bpp_algorithms import gen_random_items
from bpp_utils import bins_num
from problem_generator import get_problems

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
episode = 0

############## Hyperparameters ##############
max_episodes = 500000  # max training episodes
update_episode = 15  # update policy every n episodes (100, batch size, 越高显存占用越高)
lr_step_updates = 500  # lr decay rate (200)
save_episode = 10000  # episodes per save
input_dim = 8  # ptr-net input dim
embed_dim = 128  # lstm hidden size
lr = 0.0002  # lr (1e-3 to 1e-4, 太高容易陷入局部最优或不收敛)
betas = (0.9, 0.999)  # adam optimizer betas
gamma = 0.995  # reward discount factor (0.995 to 1.0)
K_epochs = 5  # update policy for K epochs (5 to 10, 太高容易不收敛)
eps_clip = 0.2  # clip factor
max_kl = 0.05  # kl truncate
entropy_reg = 0.005  # entropy regularization (0.001 to 0.01, 太高会难以收敛, 太低会收敛过早)
random_seed = 42
#############################################

root_path = 'D:/DRL-HH-BPP-main/PPOAgent/'
model_path = root_path + 'saved_model'
summary_path = root_path + 'summary'
test_size = 100

# 生成instance参数
instance_type = 'normal'
item_num = 200
item_std = 2
item_distribution = None


class Memory:
    """
    Record the agent experience for updates
    """

    def __init__(self):
        self.actions = []
        self.states = []
        self.masks = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.masks[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, embed_dim):
        super(ActorCritic, self).__init__()

        self.action_layer = PointerNet(input_dim=input_dim, embedding_dim=embed_dim, hidden_size=embed_dim).to(device)
        self.value_layer = PointerNet(input_dim=input_dim, embedding_dim=embed_dim, hidden_size=embed_dim).to(device)

    # actor critic
    def actor(self, state, action_mask, lengths):
        scores = self.action_layer(state, lengths)
        action_probs = masked_softmax(scores, action_mask)
        return action_probs

    def critic(self, state, action_mask, lengths):
        scores = self.value_layer(state, lengths)
        mask = action_mask.bool()
        value = scores.masked_fill(~mask, 0).sum(dim=-1, keepdim=True)
        return value

    def forward(self):
        raise NotImplementedError

    def act(self, state, mask, memory, deterministic=False):
        """
        Pass the state into actor net and return action probabilities
        :param state: embedded bin representations of shape (n, input_dim)
        :param mask: [True] * state.shape[1]
        :param memory: the memory buffer storing experiences
        :param deterministic: True - act in test mode
                              False - act in train mode
        """
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        mask = torch.from_numpy(mask).to(device).unsqueeze(0)

        length = torch.Tensor([state.shape[1]]).int()

        action_probs = self.actor(state, mask, length).squeeze(-1)
        if state.shape[1] != 1:
            action_probs = action_probs.squeeze(0)

        if deterministic and memory is not None:
            raise ValueError('Deterministic must be False in training.')

        if deterministic:
            action = torch.argmax(action_probs)
        else:
            dist = Categorical(action_probs)
            action = dist.sample()

        if memory is not None:
            memory.states.append(state)
            memory.masks.append(mask)
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action, mask, lengths):
        """
        Get the new logprobs, estimated values of the new policy
        :param state: embedded bin representations of shape (batch, n, input_dim)
        :param action: corresponding actions took in states of shape (batch, 1)
        :param mask: [True] * state.shape[1]
        :param lengths: state lengths
        """
        lengths = torch.Tensor(lengths).int()
        action_probs = self.actor(state, mask, lengths)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.critic(state, mask, lengths)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, embed_dim, lr, betas, gamma, K_epochs, eps_clip, entropy_reg, lr_step,
                 writer, load=None):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.entropy_reg = entropy_reg
        self.K_epochs = K_epochs

        self.policy = ActorCritic(embed_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.scheduler = StepLR(self.optimizer, step_size=lr_step, gamma=0.95)
        self.scaler = GradScaler()  # 减少显存占用
        self.policy_old = deepcopy(self.policy)

        self.MseLoss = nn.MSELoss()

        self.writer = writer
        if load is not None:
            self.policy.load_state_dict(torch.load(load))

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-16)
        # rewards = rewards / (rewards.std() + 1e-16)

        #### FOR PtrNet
        # get experience lengths
        lengths = [s.shape[1] for s in memory.states]
        max_length = max(lengths)
        for i, s in enumerate(memory.states):
            memory.states[i] = pad_tensor(s, size=max_length - s.shape[1])
        for i, m in enumerate(memory.masks):
            memory.masks[i] = pad_tensor(m, size=max_length - m.shape[1])

        old_states = torch.stack(memory.states).to(device).detach().squeeze(1)
        old_masks = torch.stack(memory.masks).to(device).detach().squeeze(1)
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        memory.clear_memory()

        # Optimize policy for K epochs:
        kl = 0
        losses = []
        policy_losses = []
        value_losses = []
        entropies = []

        for i in range(self.K_epochs):
            torch.cuda.empty_cache()
            # TODO: mini-batch
            # Evaluating old actions and values:
            with autocast():
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, old_masks, lengths)
                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - old_logprobs.detach())

                # Finding Surrogate Loss:
                advantages = rewards - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                policy_loss = -torch.min(surr1, surr2)
                value_loss = 0.5 * self.MseLoss(state_values, rewards)
                loss = policy_loss + value_loss - self.entropy_reg * dist_entropy

            # take gradient step
            self.scaler.scale(loss.mean()).backward()
            # self.optimizer.zero_grad()
            # loss.backward()
            # scaler.unscale_(self.optimizer)
            # torch.nn.utils.clip_grad_norm_(self.policy.action_layer.parameters(), 1.0)

            self.scaler.step(self.optimizer)
            # self.optimizer.step()
            scale = self.scaler.get_scale()
            self.scaler.update()
            skip_lr_sched = (scale != self.scaler.get_scale())
            if not skip_lr_sched:
                self.scheduler.step()

            kl += (old_logprobs.detach() - logprobs).mean().item()
            losses.append(loss.mean().item())
            policy_losses.append(policy_loss.detach().mean().item())
            value_losses.append(value_loss.detach().mean().item())
            entropies.append(dist_entropy.mean().item())
            if kl > max_kl:
                break

        self.writer.add_scalar('Policy/Kl-divergence', abs(kl), episode)
        self.writer.add_scalar('Policy/Loss', np.average(losses), episode)
        self.writer.add_scalar('Policy/Policy loss', np.average(policy_losses), episode)
        self.writer.add_scalar('Policy/Value loss', np.average(value_losses), episode)
        self.writer.add_scalar('Policy/Entropy', np.average(entropies), episode)

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


def train(test_data):
    # creating environment
    env = BPPEnv(test=False, data=None,
                 gen_mode=instance_type,
                 gen_n=item_num,
                 gen_std=item_std,
                 gen_probs=item_distribution)
    render = False

    writer = SummaryWriter(summary_path)
    memory = Memory()
    ppo = PPO(embed_dim,
              lr=lr,
              betas=betas,
              gamma=gamma,
              K_epochs=K_epochs,
              eps_clip=eps_clip,
              entropy_reg=entropy_reg,
              lr_step=lr_step_updates,
              writer=writer)

    # training loop
    bins = []
    rewards = []
    for ep in tqdm(range(1, max_episodes + 1)):
        global episode
        episode = ep

        state = env.reset()
        while not env.is_terminal():

            # Running policy_old:
            action = ppo.policy_old.act(state, env.get_action_length_mask(), memory)

            state, reward, done, _ = env.step(action)

            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            rewards.append(reward)

            if render:
                env.render()
            if done:
                bins.append(env.bf_result - bins_num(env.bins))
                break

        # update if its time
        if ep % update_episode == 0:
            writer.add_scalar('Reward/Avg reward', np.sum(rewards) / update_episode, ep)
            writer.add_scalar('Reward/Avg bin difference', np.average(bins), ep)
            ppo.update(memory)
            memory.clear_memory()
            bins.clear()
            rewards.clear()

        # stop training if avg_reward > solved_reward
        if ep % save_episode == 0:
            torch.save(ppo.policy.state_dict(), model_path + '/PPO_{}.pth'.format(ep))
            test(test_data, ppo)


def test(data, ppo=None, model_dir=None):
    # creating environment
    env = BPPEnv(test=True, data=deepcopy(data))
    render = False

    if ppo is None:
        ppo = PPO(embed_dim,
                  lr=lr,
                  betas=betas,
                  gamma=gamma,
                  K_epochs=K_epochs,
                  eps_clip=eps_clip,
                  entropy_reg=entropy_reg,
                  lr_step=lr_step_updates,
                  writer=None,
                  load=model_dir)

    # training loop
    rewards = []

    for _ in range(len(data)):

        state = env.reset()
        while not env.is_terminal():

            # Running policy_old:
            action = ppo.policy.act(state, env.get_action_length_mask(), None, True)

            state, reward, done, _ = env.step(action)

            if done:
                rewards.append(env.bf_result - bins_num(env.bins))
                break

    print('')
    print('Test complete')
    print('average:', np.average(rewards), 'std:', np.std(rewards))


if __name__ == '__main__':
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    # save dir
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    for file in os.listdir(model_path): os.remove(model_path + '/' + file)
    # summary dir
    if not os.path.exists(summary_path):
        os.mkdir(summary_path)
    for file in os.listdir(summary_path): os.remove(summary_path + '/' + file)
    # torch.autograd.set_detect_anomaly(True)

    test_data = [get_problems(mode=instance_type, n_items=item_num,
                              std=item_std, probs=item_distribution) for _ in range(test_size)]

    train(test_data)
    test(test_data, model_dir=model_path)

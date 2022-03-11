import random
import torch
from collections import namedtuple
from MCTSAgent.ResNet import EntropyLoss

Transition = namedtuple('Transition',
                        ('s', 'm', 'v', 'p'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, mode='random', model=None, gamma=1.0):
        if mode == 'random':
            return random.sample(self.memory, batch_size)
        elif mode == 'prioritized':
            if model is None:
                raise ValueError('Prioritized replay requires agent\'s model')
            ranked_samples = []
            with torch.no_grad():
                for sample in self.memory:
                    _p = torch.tensor(sample.p, device=torch.device('cuda'), dtype=torch.float)
                    p = model.predict(sample.s, sample.m, device=torch.device('cuda'))
                    m = torch.tensor(sample.m, device=torch.device('cuda'), dtype=torch.bool)
                    loss = EntropyLoss()
                    error = loss(p, _p, m)
                    rank = sample.v + gamma * error.cpu().detach().numpy()
                    ranked_samples.append([sample, rank])
            ranked_samples.sort(key=lambda x: x[1], reverse=True)

            sampled = [sample[0] for sample in ranked_samples[:batch_size]]
            random.shuffle(sampled)
            return sampled
        else:
            raise ValueError('Invalid sample mode')

    def __len__(self):
        return len(self.memory)

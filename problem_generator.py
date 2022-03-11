import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

s_lower_bound = 10
s_upper_bound = 60
item_types = np.array([i for i in range(10, 61, 5)])
capacity = 100


def get_problems(mode: str, n_items: int, std: float = None, probs=None):
    if mode == 'uniform':
        items = np.random.choice(item_types, size=n_items)
    elif mode == 'normal':
        if std is None:
            raise ValueError('Require std for item generation in this mode')
        x = np.arange(-5, 6)
        xU, xL = x + 0.5, x - 0.5
        prob = ss.norm.cdf(xU, scale=std) - ss.norm.cdf(xL, scale=std)
        prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
        items = np.random.choice(item_types, size=n_items, p=prob)
    elif mode == 'other':
        if probs is None:
            raise ValueError('Require customized probs for item generation in this mode')
        probs = np.array(probs)
        probs = probs / probs.sum()
        items = np.random.choice(item_types, size=n_items, p=probs)
    else:
        raise ValueError('Invalid mode')
    return items / capacity


if __name__ == '__main__':
    # 生成均匀分布
    items = get_problems(mode='uniform', n_items=1000)
    print(np.unique(items, return_counts=True))
    # 生成正态分布
    items = get_problems(mode='normal', n_items=1000, std=2)
    print(np.unique(items, return_counts=True))
    # 生成自定概率分布
    items = get_problems(mode='other', n_items=1000,
                         probs=[0.33, 0.30, 0.27, 0.24, 0.21, 0.18, 0.15, 0.12, 0.09, 0.06, 0.03])
    print(np.unique(items, return_counts=True))

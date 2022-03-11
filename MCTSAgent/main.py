from MCTSAgent.State import *
from MCTSAgent.Agent import *
from MCTSAgent.ResNet import *
from MCTSAgent.Policy import *
from bpp_algorithms import gen_triplet, gen_random_items, gen_items, best_fit
# from reader import read_instances, sample_instances
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import random
import numpy as np
import time
import os.path
from multiprocessing import Pool

"""
    Pipeline - MCTS for Bin Packing:
    1. Initialize state
    2. Make sampler's move (give the next item)
    3. Agent observe the environment
    4. If multiple actions available, run MCTS with random item sampler as the adversary
    5. If only one action is legal, take the action
    6. Store experience in memory if multiple actions available
    7. Train the policy network
"""

# saved model
base_dir = './MCTSAgent/'
model_path = base_dir + 'saved_model900_bf_roll_bf-d/'
summary_path = base_dir + 'summary/'

# cuda device
search_device = torch.device('cuda')
update_device = torch.device('cpu')
# parallel execution
n_processes = 10

train_episodes = 10000
batch_size = 64
learning_rate = 0.0005
exploration = 2
sampler_actions = 50

# search iterations
train_search_iterations = 500
test_search_iterations = 1000
# adversary_search_iterations = 500
buffer_size = 250


def train():
    agent.model.train()
    last_update_rewards = []
    rank_buffer = []

    for i in tqdm(range(train_episodes)):
        # root parallel search
        pool = Pool(processes=n_processes)
        res = pool.starmap(run, [(deepcopy(agent), gen_random_items(20), 'train', 'bf') for _ in range(n_processes)])
        pool.close()
        pool.join()

        # add to rank buffer (for ranked reward)
        # for r in res:
        #     for _, _, _v, _ in r:
        #         update_rank(rank_buffer, _v)
        #         break

        for r in res:
            reward_noted = False
            # reward = 0
            for s, m, _v, _p in r:
                if not reward_noted:
                    last_update_rewards.append(_v)
                    # reward = get_ranked_reward(rank_buffer, _v, 0.5)
                    reward_noted = True
                agent.memory.push(s, m, _v, _p)

        # update
        if i % 5 == 0 and i >= 10:
            writer.add_scalar('Reward/Avg. reward', np.average(last_update_rewards), i)
            last_update_rewards.clear()

        if i >= 10:
            for _ in range(5):
                agent.experience(writer=writer)

        # save model
        if i % 100 == 0:
            # save model
            torch.save(agent.model.state_dict(), model_path + 'model-' + str(i) + '.pt')

        # evaluate in every 100 epochs
        if i % 200 == 0 and i > 0:
            random_results = test_parallel(agent_=agent, instances=test_data, metric='bf')
            print('')
            print('====== Evaluation random %s ======' % i)
            print('Update:', i, ', result:', np.unique(random_results, return_counts=True),
                  ', average:', np.mean(random_results))
            time.sleep(0.1)


def test_parallel(agent_, instances, metric, num_workers=10):
    # root parallel search
    pool = Pool(processes=num_workers)
    res = pool.starmap(run, [(deepcopy(agent_), i, 'test', metric) for i in instances])
    pool.close()
    pool.join()
    return res


def run(agent_, items, mode, metric='bf'):
    agent_.model = agent_.model.to(device=search_device)
    if mode == 'train':
        agent_.set_search(iteration=train_search_iterations)
        agent_.model.train()
    elif mode == 'test':
        if not isinstance(agent_, BestFitAgent):
            agent_.set_search(iteration=test_search_iterations)
            agent_.model.eval()

    states = list()
    masks = list()
    real_ps = list()


    state = State(items=items)
    while not state.terminal():
        # opponent move
        if metric == 'adverse':
            mcts = MCTS(agent_, exploration=1, rollout_policy=random_policy)
            mcts.set_root(state)
            for m in range(adversary_search_iterations):
                mcts.execute()

            action, _ = mcts.search()
            state = state.child_state(action=action)
        elif metric == 'bf':
            state = state.child_state(action=state.items[0])

        # agent move
        action, _, real_p = agent_.observe(state)
        # add experience
        if action is not None and mode == 'train':
            states.append(state.state())
            masks.append(state.mask)
            real_ps.append(real_p)

        state = agent_.act(state=state, action=action)

    # end episode
    _v = state.reward

    if mode == 'train':
        return zip(states, masks, [_v] * len(states), real_ps)
    elif mode == 'test':
        result = cb.best_fit([], items)
        return result - bins_num(state.bins)
    else:
        raise ValueError('Invalid run mode')


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # save dir
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    # summary dir
    if not os.path.exists(summary_path):
        os.mkdir(summary_path)

    test_data = [gen_random_items(20) for _ in range(500)]
    # summary writer
    writer = SummaryWriter(summary_path)

    model = ResNet(lr=learning_rate)
    model.load_state_dict(torch.load(model_path+'model-900.pt'))
    agent = Agent(model=model, exploration=exploration, batch_size=batch_size,
                  search_iteration=train_search_iterations)

    # train()

    results = test_parallel(agent_=agent, instances=test_data, metric='bf')
    print('\033[1;31m====== END ======\033[0m')
    print('results:', results)
    print('average:', np.average(results))

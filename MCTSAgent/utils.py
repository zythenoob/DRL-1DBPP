import math
import random


def update_rank(buffer, terminal_reward):
    # update rank buffer
    # if terminal_reward not in self.rank_buffer:
    buffer.append(terminal_reward)
    # buffer.sort(reverse=False)
    if len(buffer) > 500:
        buffer.pop(0)

    # store rank buffer
    with open('temp_rank.txt', 'w') as f:
        f.writelines(map(lambda x: str(x) + '\n', buffer))


def get_ranked_reward(buffer, reward, rank=0.75):
    # get rank
    if len(buffer) <= 1:
        return 1

    sorted_buffer = buffer.copy()
    sorted_buffer.sort(reverse=False)

    alpha_index = math.floor(len(sorted_buffer) * rank) - 1
    reward_alpha = sorted_buffer[alpha_index]

    if reward > reward_alpha or reward == 1.0:
        return 1
    elif reward < reward_alpha:
        return -1
    elif reward == reward_alpha and reward < 1.0:
        return -1 if random.randint(0, 1) == 0 else 1

from bpp_utils import *
import c_bpp


def best_fit(bins_in, items):
    bins = [b.copy() for b in bins_in]
    for i in items:
        pack = -1
        best_fit_space = -1
        for b in range(len(bins)):
            if np.sum(bins[b]) + i <= 1:
                if best_fit_space == -1 or 1 - np.sum(bins[b]) < best_fit_space:
                    pack = b
                    best_fit_space = 1 - np.sum(bins[b])

        if pack == -1:
            bins.append([i])
        else:
            bins[pack].append(i)

    return bins_num(bins)

# MBS
def relaxed_mbs_predict(seed, queue, relax):
    bin = seed.copy()
    items = queue.copy()
    items.sort(reverse=True)
    if len(bin) == 0 and len(items) > 0:
        bin.append(items[0])
        items.pop(0)
    prediction, pred_bin, _ = mbs_one_packing(bin, items, np.sum(bin), relax, bin, [])
    return round(prediction, 5), pred_bin


def mbs_one_packing(bin, items, best, relax, best_bin, pack_history):
    for item in items:
        # print('')
        # print('current:', bin)
        # print("checking:", item)
        if 1 - np.sum(bin) < items[-1]:
            break
        if item + np.sum(bin) <= 1.0:
            # print('can pack')
            bin.append(item)
            # print('history:', pack_history)
            if bin not in pack_history:
                # print('not in history')
                pack_history.append(bin.copy())
                best, best_bin, pack_history = mbs_one_packing(bin, pop_items(items, [item]),
                                                               best, relax, best_bin, pack_history)
                bin.pop()
            else:
                # print('in history')
                bin.pop()
                continue
            if best > relax:
                return best, best_bin, pack_history
    if np.sum(bin) > best or (np.sum(bin) == best and len(bin) < len(best_bin)):
        best = np.sum(bin)
        # print('find best:', best)
        best_bin = bin.copy()
    return best, best_bin, pack_history


def relaxed_mbs_fit(seeds, queue, relax, min_item):
    bins = [b.copy() for b in seeds]
    items = queue.copy()
    items.sort(reverse=True)
    # print('fit start')
    if len(bins) == 0 and len(items) > 0:
        bins.append([items[0]])
        items.pop(0)

    # print('items:', items)
    # print('bins_num:', bins_num)
    # print('fit loop')
    while len(items) > 0:
        # print('')
        for i in range(len(bins)):
            if np.sum(bins[i]) < 1 - min_item:
                # print('fit bin:', i)
                _, pred_bin = relaxed_mbs_predict(bins[i], items, relax)
                packed_items = pred_bin[len(bins[i]):]
                bins[i] = pred_bin
                # print('fitted:', pred_bin)
                items = pop_items(items, packed_items)

        if len(items) == 0:
            break

        # print('new bin, len items:', len(items))
        _, new_bin = relaxed_mbs_predict([items[0]], items[1:], relax)
        bins.append(new_bin)
        items = pop_items(items, new_bin)
        # print('fitted:', new_bin)

    # print('pred bins_num:', bins_num)
    return bins


def pop_items(items, pop):
    items = items.copy()
    for p in pop:
        items.remove(p)
    return items


def generate_triplet_files(num_of_instances, n_bins):
    with open('triplet_list.txt', 'w') as f_list:
        for n in range(num_of_instances):
            filename = 'triplet_' + str(n) + '.txt'
            with open('triplet_data/' + filename, 'w') as f_instance:
                f_instance.write(str(n_bins * 3) + '\n')
                f_instance.write('100\n')
                items = []
                for i in range(n_bins):
                    item1 = 0
                    item2 = 0
                    while item1 == item2:
                        item1 = np.random.randint(1, 98)
                        item2 = np.random.randint(item1, 98)
                    items.extend([str(item1) + '\n', str(item2-item1) + '\n', str(100-item2) + '\n'])
                print(items)
                random.shuffle(items)
                f_instance.writelines(items)
            f_list.write(filename+'\n')


def gen_random_items(n):
    return [np.random.randint(1, 100) / 100 for _ in range(n)]


def gen_variable_length_items(lower, upper):
    return [np.random.randint(1, 100) / 100 for _ in range(np.random.randint(lower, upper))]


def gen_triplet(n_bins):
    items = []
    for i in range(n_bins):
        item1 = 0
        item2 = 0
        while item1 == item2:
            item1 = np.random.randint(1, 98)
            item2 = np.random.randint(item1, 98)
        items.extend([item1/100, (item2 - item1)/100, (100 - item2)/100])
    random.shuffle(items)
    return items


def gen_tuple(n_bins):
    items = []
    for i in range(n_bins):
        item1 = np.random.randint(1, 100)
        items.extend([item1/100, (100 - item1)/100])
    random.shuffle(items)
    return items


def gen_items(n, m):
    items = gen_triplet(n) + gen_tuple(m)
    random.shuffle(items)
    return items


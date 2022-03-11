import numpy as np
import random
import matplotlib.pyplot as plt
import math


# return the number of non-empty bins_num
def bins_num(bins):
    b = [i for i in bins if len(i) > 0]
    return len(b)


# return the overall space utilization rate of non-empty bins_num
def utilization(bins):
    if len(bins) > 0:
        uti_bins = [np.sum(b) for b in bins if len(b) > 0]
        if len(uti_bins) == 0:
            return 0
        else:
            return round(np.average(uti_bins), 6)
    else:
        return 0


# generate a list of items by histogram
def hist2items(hist, upper=1.0):
    pred = []
    unit_item_size = 1 / len(hist)
    for i in range(len(hist)):
        item = round((unit_item_size * i + unit_item_size * (i + 1)) / 2, 2) * upper
        pred = pred + [item] * hist[i]

    random.shuffle(pred)
    # print(pred)
    return pred

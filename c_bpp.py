from ctypes import *
import numpy as np
import os
import time
from numpy.ctypeslib import ndpointer

_2darray = ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS')
_1darray = ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
_1darray_int = ndpointer(dtype=np.uintp, ndim=1, flags='C_CONTIGUOUS')

os.chdir("D:/DRL-HH-BPP-main/lib")
libc = cdll.LoadLibrary("D:/DRL-HH-BPP-main/lib/c_bpp.dll")

libc.c_online_mbs.argtypes = [_2darray, c_int, c_int, _1darray, c_int, _1darray_int, c_double, c_int]
libc.c_online_mbs.restype = c_int

libc.c_offline_mbs.argtypes = [_2darray, c_int, c_int, _1darray, c_int, c_double, c_double]
libc.c_offline_mbs.restype = c_int

libc.c_mbs_demo.argtypes = [_2darray, c_int, c_int, c_double, _1darray, c_int, c_double, c_double]
libc.c_mbs_demo.restype = c_int

libc.c_bf.argtypes = [_2darray, c_int, c_int, _1darray, c_int]
libc.c_bf.restype = c_int

libc.c_bf_step.argtypes = [_2darray, c_int, c_int, c_double, c_double]
libc.c_bf_step.restype = c_int

libc.c_wf_step.argtypes = [_2darray, c_int, c_int, c_double, c_double]
libc.c_wf_step.restype = c_int

libc.c_ff_step.argtypes = [_2darray, c_int, c_int, c_double]
libc.c_ff_step.restype = c_int

bin_max_width = 50


def fill_bins(bins):
    bin_filled = np.zeros((len(bins), bin_max_width))
    if len(bins) > 0:
        for i, b in enumerate(bins):
            bin_filled[i, :len(b)] = np.array(b)
    return bin_filled


def online_mbs(bins, items, histogram, min_item, step=0):
    bins = [b for b in bins if len(b) > 0]
    bin_filled = fill_bins(bins)

    c_in = bin_filled.astype(np.float64)
    c_h = c_int(len(bins))
    c_w = c_int(bin_max_width)
    c_items = np.array(items).astype(np.float64)
    c_n_items = len(items)
    c_hist = np.array(histogram).astype(np.uintp)
    c_min = c_double(min_item)
    c_step = c_int(step)
    c_r = libc.c_online_mbs(c_in, c_h, c_w, c_items, c_n_items, c_hist, c_min, c_step)
    return c_r


def offline_mbs(bins, items, min_item, relax):
    bins = [b for b in bins if len(b) > 0]
    bin_filled = fill_bins(bins)

    c_in = bin_filled.astype(np.float64)
    c_h = c_int(len(bins))
    c_w = c_int(bin_max_width)
    c_items = np.array(items).astype(np.float64)
    c_n_items = len(items)
    c_min = c_double(min_item)
    c_relax = c_double(relax)
    c_r = libc.c_offline_mbs(c_in, c_h, c_w, c_items, c_n_items, c_min, c_relax)
    return c_r


def mbs_demo(bins, item, items, min_item, relax=1.0):
    bins = [b.copy() for b in bins if len(b) > 0]
    bins_dict = dict(zip(range(len(bins)), bins))
    bins.sort(reverse=True, key=lambda x: np.sum(x))
    bins_sorted_dict = dict(zip(range(len(bins)), bins))
    bin_filled = fill_bins(bins)

    c_in = bin_filled.astype(np.float64)
    c_h = c_int(len(bins))
    c_w = c_int(bin_max_width)
    c_item = c_double(item)
    c_items = np.array(items).astype(np.float64)
    c_n_items = len(items)
    c_min = c_double(min_item)
    c_relax = c_double(relax)
    c_r = libc.c_mbs_demo(c_in, c_h, c_w, c_item, c_items, c_n_items, c_min, c_relax)

    if not c_r == -1:
        target = bins_sorted_dict.get(c_r)
        for i, b in enumerate(bins_dict.items()):
            if b[1] == target:
                c_r = i
    return c_r


def best_fit(bins, items):
    bins = [b.copy() for b in bins if len(b) > 0]
    bin_filled = fill_bins(bins)

    c_in = bin_filled.astype(np.float64)
    c_h = c_int(len(bins))
    c_w = c_int(bin_max_width)
    c_items = np.array(items).astype(np.float64)
    c_n_items = len(items)
    c_r = libc.c_bf(c_in, c_h, c_w, c_items, c_n_items)
    return c_r


def bf_step(bins, item, thresh=1.0):
    bin_filled = fill_bins(bins)

    c_in = bin_filled.astype(np.float64)
    c_h = c_int(len(bins))
    c_w = c_int(bin_max_width)
    c_item = c_double(item)
    c_thresh = c_double(thresh)
    c_r = libc.c_bf_step(c_in, c_h, c_w, c_item, c_thresh)
    return c_r


def wf_step(bins, item, thresh=1.0):
    bin_filled = fill_bins(bins)

    c_in = bin_filled.astype(np.float64)
    c_h = c_int(len(bins))
    c_w = c_int(bin_max_width)
    c_item = c_double(item)
    c_thresh = c_double(thresh)
    c_r = libc.c_wf_step(c_in, c_h, c_w, c_item, c_thresh)
    return c_r


def ff_step(bins, item):
    bin_filled = fill_bins(bins)

    c_in = bin_filled.astype(np.float64)
    c_h = c_int(len(bins))
    c_w = c_int(bin_max_width)
    c_item = c_double(item)
    c_r = libc.c_ff_step(c_in, c_h, c_w, c_item)
    return c_r


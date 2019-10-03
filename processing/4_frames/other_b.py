import sys
import argparse, random
from compress_pickle import load, dump
from datetime import datetime as dt
from sklearn.utils.extmath import cartesian
import numpy as np, pandas as pd

sys.path.append('repo/')
from constants import *
from utils import *

sys.path.append('repo/processing/')
from processing_utils import *


def multiply_indices(s):
    # initialize arrays
    k = len(s.index.names)
    arrays = np.zeros((s.sum(),k+1), dtype=np.int64)
    count = 0
    # outer loop: range length
    for i in range(1, max(s)+1):
        index = s.index[s == i].values
        if len(index) == 0:
            continue
        # cartesian product of existing level(s) and period
        if k == 1:
            f = lambda x: cartesian([[x], list(range(i))])
        else:
            f = lambda x: cartesian([[e] for e in x] + [list(range(i))])
        # inner loop: rows of period
        for j in range(len(index)):
            arrays[count:count+i] = f(index[j])
            count += i
    # convert to multi-index
    return pd.MultiIndex.from_arrays(np.transpose(arrays), 
        names=s.index.names + ['period'])


def parse_days(diff, t0, t1):
    # count of arrivals by day
    days = diff.dt.days.rename('period').to_frame().assign(count=1)
    days = days.groupby(['lstg', 'period']).sum().squeeze().astype(np.uint8)
    # end of listings
    T1 = int((pd.to_datetime(END) - pd.to_datetime(START)).total_seconds())
    t1[t1 > T1] = T1
    end = (pd.to_timedelta(t1 - t0, unit='s').dt.days).rename('period')
    # create multi-index from end stamps
    idx = multiply_indices(end+1)
    # expand to new index and return
    return days.reindex(index=idx, fill_value=0).sort_index()


def get_y_arrival(lstgs, threads):
    d = {}
    # time_stamps
    t0 = lstgs.start_date * 24 * 3600
    t1 = lstgs.end_time
    diff = pd.to_timedelta(threads.clock - t0, unit='s')
    # append arrivals to end stamps
    d['days'] = parse_days(diff, t0, t1)
    # create other outcomes
    d['loc'] = threads.byr_us.rename('loc')
    d['hist'] = threads.byr_hist.rename('hist')
    d['bin'] = threads.bin
    sec = ((diff.dt.seconds[threads.bin == 0] + 0.5) / (24 * 3600 + 1))
    d['sec'] = sec.rename('sec')
    return d


if __name__ == "__main__":
    # partition number from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num

    # partition
    partitions = load(PARTS_DIR + 'partitions.gz')
    part = list(partitions.keys())[num-1]
    idx = partitions[part]
    path = lambda name: PARTS_DIR + part + '/' + name + '.gz'

    # load data and 
    lstgs = pd.read_csv(CLEAN_DIR + 'listings.csv').drop(
        ['title', 'flag'], axis=1).set_index('lstg').reindex(index=idx)
    threads = load_frames('threads').reindex(index=idx, level='lstg')

    # outcomes for arrival model
    print('Creating arrival model outcome variables')
    y_arrival = get_y_arrival(lstgs, threads)
    for k, v in y_arrival.items():
        dump(v, path('y_arrival_' + k))
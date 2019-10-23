import sys, os
sys.path.append('repo/')
sys.path.append('repo/processing/')
import argparse, pickle
import numpy as np, pandas as pd
from constants import *


if __name__ == "__main__":
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num

    # load data
    print('Loading data')
    d = pickle.load(open(CHUNKS_DIR + '%d.pkl' % num, 'rb'))
    T = d['threads']
    d = pickle.load(open(CHUNKS_DIR + '%d_events_lstgs.pkl' % num, 'rb'))
    events, lstgs = [d[k] for k in ['events', 'lstgs']]

    # split off threads dataframe
    events = events.join(T[['byr_hist', 'byr_us']])
    threads = events[['clock', 'byr_us', 'byr_hist', 'bin']].xs(
        1, level='index')
    events = events.drop(['byr_us', 'byr_hist', 'bin'], axis=1)

    # exclude current thread from byr_hist
    threads['byr_hist'] -= (1-threads.bin) 

    # write chunk
    print("Writing chunk")
    chunk = {'events': events, 'lstgs': lstgs, 'threads': threads}
    pickle.dump(chunk, open(CHUNKS_DIR + '%d_frames.pkl' % num, 'wb'))
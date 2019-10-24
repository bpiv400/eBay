import sys
import argparse, random
from compress_pickle import load, dump
from datetime import datetime as dt
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing.processing_utils import *


def get_y_delay(delay, censored, role):
    # restrict to role indices
    s = delay[delay.index.isin(IDX[role], level='index')]
    c = censored.reindex(index=s.index)
    # expirations
    exp = s >= MAX_DELAY[role]
    if role == 'byr':
        exp.loc[exp.index.isin([7], level='index')] = s >= MAX_DELAY['slr']
    # interval of offer arrivals and censoring
    arrival = (s[~exp & ~c] / INTERVAL[role]).astype('uint16').rename('arrival')
    cens = (s[~exp & c] / INTERVAL[role]).astype('uint16').rename('cens')
    # initialize output dataframe with arrivals
    df = arrival.to_frame().assign(count=1).set_index(
        'arrival', append=True).squeeze().unstack(
        fill_value=0).reindex(index=s.index, fill_value=0)
    # vector of censoring thresholds
    v = (arrival+1).append(cens, verify_integrity=True).reindex(
        s.index, fill_value=INTERVAL_COUNTS[role])
    if role == 'byr':
        mask = v.index.isin([7], level='index') & (v > INTERVAL_COUNTS['byr_7'])
        v.loc[mask] = INTERVAL_COUNTS['byr_7']
    # replace censored observations with -1
    for i in range(INTERVAL_COUNTS[role]):
        df[i] -= (i >= v).astype('int8')
    return df


if __name__ == "__main__":
    # partition number from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num-1

    # partition
    part = PARTITIONS[num]
    idx, path = get_partition(part)

    # load events
    events = load_frames('events')[['clock', 'censored']].reindex(
        index=idx, level='lstg')
    censored = events.censored

    # calculate delay
    clock = events.clock.unstack()
    delay = pd.DataFrame(index=clock.index)
    for i in range(2, 8):
        delay[i] = clock[i] - clock[i-1]
        delay.loc[delay[i] == 0, i] = np.nan  # remove auto responses
    delay = delay.rename_axis('index', axis=1).stack().astype('int64')

    # byr delay
    print('y_delay_byr')
    y_delay_byr = get_y_delay(delay, censored, 'byr')
    dump(y_delay_byr, path('y_delay_byr'))

    # slr delay
    y_delay_slr = get_y_delay(delay, censored, 'slr')
    dump(y_delay_slr, path('y_delay_slr'))
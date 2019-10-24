import sys
import argparse, random
from compress_pickle import load, dump
from datetime import datetime as dt
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing.processing_utils import *


def get_y_arrival(lstgs, threads):
    # time_stamps
    t0 = lstgs.start_date * 24 * 3600
    diff = pd.to_timedelta(threads.start_time - t0, unit='s')
    end = pd.to_timedelta(lstgs.end_time - t0, unit='s')
    # convert to hours
    diff = (diff.dt.total_seconds() // 3600).astype('uint16')
    end = (end.dt.total_seconds() // 3600).astype('uint16')
    # censor to first 31 days
    diff = diff.loc[diff < MAX_DAYS * 24]
    end.loc[end >= MAX_DAYS * 24] = MAX_DAYS * 24 - 1
    # count of arrivals by hour
    hours = diff.rename('period').to_frame().assign(count=1)
    arrivals = hours.groupby(['lstg', 'period']).sum()
    arrivals = arrivals.squeeze().astype('int8')
    # initialize output dataframe
    N = np.max(end)+1
    df = pd.DataFrame(0, index=end.index, dtype='int8',
        columns=range(N))
    # fill in arrivals and censored times
    for i in range(N):
        value = arrivals.xs(i, level='period').reindex(
            index=end.index, fill_value=0)
        value -= (end < i).astype('int8')
        df[i] = value
    return df


if __name__ == "__main__":
    # partition number from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num-1

    # partition
    part = PARTITIONS[num]
    idx, path = get_partition(part)

    # load data
    lstgs = load(CLEAN_DIR + 'listings.gz')
    lstgs = lstgs[['start_date', 'end_time']].reindex(index=idx)
    threads = load(CLEAN_DIR + 'threads.gz').reindex(
        index=idx, level='lstg')

    # thread variables
    print('x_thread')
    dump(threads[['byr_hist']], path('x_thread'))

    # outcomes for arrival model
    print('Creating arrival model outcome variables')
    y_arrival = get_y_arrival(lstgs, threads)
    dump(y_arrival, path('y_arrival'))
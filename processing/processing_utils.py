import argparse
import numpy as np
import pandas as pd
from compress_pickle import load
from utils import slr_norm, byr_norm, extract_clock_feats
from constants import START, PARTITIONS, PARTS_DIR, MAX_DELAY, IDX, \
    BYR_PREFIX, SLR_PREFIX, DAY, HOLIDAYS
from featnames import HOLIDAY, DOW_PREFIX, TIME_OF_DAY, AFTERNOON, \
    CLOCK_FEATS, BYR_HIST, THREAD_COUNT


# function to load file from partitions directory
def load_file(part, x):
    return load(PARTS_DIR + '{}/{}.gz'.format(part, x))


def extract_day_feats(seconds):
    """
    Creates clock features from timestamps.
    :param seconds: seconds since START.
    :return: tuple of time_of_day sine transform and afternoon indicator.
    """
    clock = pd.to_datetime(seconds, unit='s', origin=START)
    df = pd.DataFrame(index=clock.index)
    df[HOLIDAY] = clock.dt.date.astype('datetime64').isin(HOLIDAYS)
    for i in range(6):
        df[DOW_PREFIX + str(i)] = clock.dt.dayofweek == i
    return df


def collect_date_clock_feats(seconds):
    """
    Combines date and clock features.
    :param seconds: seconds since START.
    :return: dataframe of date and clock features.
    """
    df = extract_day_feats(seconds)
    df[TIME_OF_DAY], df[AFTERNOON] = extract_clock_feats(seconds)
    assert list(df.columns) == CLOCK_FEATS
    return df


def input_partition():
    """
    Parses command line input for partition name.
    :return part: string partition name.
    """
    parser = argparse.ArgumentParser()

    # partition
    parser.add_argument('--part', required=True, type=str,
                        choices=PARTITIONS, help='partition name')
    return parser.parse_args().part


def get_days_delay(clock):
    """
    Calculates time between successive offers.
    :param clock: dataframe with index ['lstg', 'thread'], 
        turn numbers as columns, and seconds since START as values
    :return days: fractional number of days between offers.
    :return delay: time between offers as share of MAX_DELAY.
    """
    # initialize output dataframes in wide format
    days = pd.DataFrame(0., index=clock.index, columns=clock.columns)
    delay = pd.DataFrame(0., index=clock.index, columns=clock.columns)

    # for turn 1, days and delay are 0
    for i in range(2, 8):
        days[i] = clock[i] - clock[i - 1]
        delay[i] = days[i] / MAX_DELAY[i]
    # no delay larger than 1
    assert delay.max().max() <= 1

    # reshape from wide to long
    days = days.rename_axis('index', axis=1).stack() / DAY
    delay = delay.rename_axis('index', axis=1).stack()

    return days, delay


def round_con(con):
    """
    Round concession to nearest percentage point.
    :param con: pandas series of unrounded concessions.
    :return: pandas series of rounded concessions.
    """
    rounded = np.round(con, decimals=2)
    rounded.loc[(rounded == 1) & (con < 1)] = 0.99
    rounded.loc[(rounded == 0) & (con > 0)] = 0.01
    return rounded


def get_con(offers, start_price):
    # compute concessions
    con = pd.DataFrame(index=offers.index)
    con[1] = offers[1] / start_price
    con[2] = (offers[2] - start_price) / (offers[1] - start_price)
    for i in range(3, 8):
        con[i] = (offers[i] - offers[i - 2]) / (offers[i - 1] - offers[i - 2])

    # stack into series
    con = con.rename_axis('index', axis=1).stack()

    # first buyer concession should be greater than 0
    assert con.loc[con.index.isin([1], level='index')].min() > 0

    # round concessions
    rounded = round_con(con)

    return rounded


def get_norm(con):
    """
    Calculate normalized concession from rounded concessions.
    :param con: pandas series of rounded concessions.
    :return: pandas series of normalized concessions.
    """
    df = con.unstack()
    norm = pd.DataFrame(index=df.index, columns=df.columns)
    norm[1] = df[1]
    norm[2] = df[2] * (1 - norm[1])
    for i in range(3, 8):
        if i in IDX[BYR_PREFIX]:
            norm[i] = byr_norm(con=df[i],
                               prev_byr_norm=norm[i - 2],
                               prev_slr_norm=norm[i - 1])
        elif i in IDX[SLR_PREFIX]:
            norm[i] = slr_norm(con=df[i],
                               prev_byr_norm=norm[i - 1],
                               prev_slr_norm=norm[i - 2])
    return norm.rename_axis('index', axis=1).stack().astype('float64')


def init_x(part, idx):
    x = load_file(part, 'x_lstg')
    x = {k: v.reindex(index=idx, level='lstg').astype('float32') for k, v in x.items()}
    return x


def get_x_thread(threads, idx):
    # initialize x_thread as copy
    x_thread = threads.copy()

    # byr_hist as a decimal
    x_thread.loc[:, BYR_HIST] = x_thread.byr_hist.astype('float32') / 10

    # thread count, including current thread
    x_thread[THREAD_COUNT] = x_thread.index.get_level_values(level='thread')

    # reindex to create x_thread
    x_thread = pd.DataFrame(index=idx).join(x_thread)

    return x_thread.astype('float32')

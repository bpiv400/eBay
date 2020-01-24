import argparse
import numpy as np, pandas as pd
from utils import extract_clock_feats
from constants import MINUTE, HOUR, DAY, MONTH, HOLIDAYS, PARTITIONS, MAX_DELAY, \
    IDX, BYR_PREFIX, SLR_PREFIX, START
from featnames import HOLIDAY, DOW_PREFIX, TIME_OF_DAY, AFTERNOON, CLOCK_FEATS, MONTHS_SINCE_LSTG
from utils import slr_norm, byr_norm


def extract_day_feats(seconds):
    '''
    Creates clock features from timestamps.
    :param seconds: seconds since START.
    :return: tuple of time_of_day sine transform and afternoon indicator.
    '''
    clock = pd.to_datetime(seconds, unit='s', origin=START)
    df = pd.DataFrame(index=clock.index)
    df[HOLIDAY] = clock.dt.date.astype('datetime64').isin(HOLIDAYS)
    for i in range(6):
        df[DOW_PREFIX + str(i)] = clock.dt.dayofweek == i
    return df


def collect_date_clock_feats(seconds):
    '''
    Combines date and clock features.
    :param seconds: seconds since START.
    :return: dataframe of date and clock features.
    '''
    df = extract_day_feats(seconds)
    df[TIME_OF_DAY], df[AFTERNOON] = extract_clock_feats(seconds)
    assert all(df.columns == CLOCK_FEATS)
    return df


def input_partition():
    """
    Parses command line input for partition name.
    :return part: string partition name.
    """
    parser = argparse.ArgumentParser()

    # partition
    parser.add_argument('--part', required=True, type=str, help='partition name')
    part = parser.parse_args().part

    # error checking
    if part not in PARTITIONS:
        raise RuntimeError('part must be one of: {}'.format(PARTITIONS))

    return part


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
        days[i] = clock[i] - clock[i-1]
        if i in [2, 4, 6, 7]: # byr has 2 days for last turn
            delay[i] = days[i] / MAX_DELAY[SLR_PREFIX]
        elif i in [3, 5]:   # ignore byr arrival and last turn
            delay[i] = days[i] / MAX_DELAY[BYR_PREFIX]
    # no delay larger than 1
    assert delay.max().max() <= 1

    # reshape from wide to long
    days = days.rename_axis('index', axis=1).stack() / DAY
    delay = delay.rename_axis('index', axis=1).stack()

    return days, delay


def get_norm(con):
    '''
    Calculate normalized concession from rounded concessions.
    :param con: pandas series of rounded concessions.
    :return: pandas series of normalized concessions.
    '''
    df = con.unstack()
    norm = pd.DataFrame(index=df.index, columns=df.columns)
    norm[1] = df[1]
    norm[2] = df[2] * (1-norm[1])
    for i in range(3, 8):
        if i in IDX[BYR_PREFIX]:
            norm[i] = byr_norm(con=df[i], 
                prev_byr_norm=norm[i-2],
                prev_slr_norm=norm[i-1])
        elif i in IDX[SLR_PREFIX]:
            norm[i] = slr_norm(con=df[i],
                prev_byr_norm=norm[i-1],
                prev_slr_norm=norm[i-2])
    return norm.rename_axis('index', axis=1).stack().astype('float64')

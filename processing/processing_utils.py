import argparse
import numpy as np
import pandas as pd
from compress_pickle import load
from utils import slr_norm, byr_norm, extract_clock_feats, is_split
from constants import *
from featnames import *


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


def process_sim_offers(df):
    # clock features
    df = df.join(collect_date_clock_feats(df.clock))
    # days and delay
    df[DAYS], df[DELAY] = get_days_delay(df.clock.unstack())
    # concession as a decimal
    df.loc[:, CON] /= 100
    # indicator for split
    df[SPLIT] = df[CON].apply(lambda x: is_split(x))
    # total concession
    df[NORM] = get_norm(df[CON])
    # reject auto and exp are last
    df[REJECT] = df[CON] == 0
    df[AUTO] = (df[DELAY] == 0) & df.index.isin(IDX[SLR_PREFIX], level='index')
    df[EXP] = df[DELAY] == 1
    # reorder columns to match observed
    df = df.loc[:, ALL_OFFER_FEATS]
    return df


def process_sim_threads(part, df):
    # convert clock to months_since_lstg
    df = df.join(load_file(part, 'lookup').start_time)
    df[MONTHS_SINCE_LSTG] = (df.clock - df.start_time) / MONTH
    df = df.drop(['clock', 'start_time'], axis=1)
    # reorder columns to match observed
    df = df.loc[:, [MONTHS_SINCE_LSTG, BYR_HIST]]
    return df


def concat_sim_chunks(part):
    """
    Loops over simulations, concatenates dataframes.
    :param part: string name of partition.
    :return: concatentated and sorted threads and offers dataframes.
    """
    # collect chunks
    threads, offers = [], []
    for i in range(1, SIM_CHUNKS + 1):
        sim = load(ENV_SIM_DIR + '{}/discrim/{}.gz'.format(part, i))
        threads.append(sim['threads'])
        offers.append(sim['offers'])

    # concatenate
    threads = pd.concat(threads, axis=0).sort_index()
    offers = pd.concat(offers, axis=0).sort_index()

    # drop censored offers
    offers = offers.loc[~offers[CENSORED], :]
    offers.drop(CENSORED, axis=1, inplace=True)

    # conform to observed inputs
    threads = process_sim_threads(part, threads)
    offers = process_sim_offers(offers)

    return threads, offers

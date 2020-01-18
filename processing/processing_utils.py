import argparse
import numpy as np, pandas as pd
from constants import MINUTE, HOUR, DAY, MONTH, HOLIDAYS, PARTITIONS, MAX_DELAY, IDX
from featnames import CLOCK_FEATS, MONTHS_SINCE_LSTG


def extract_day_feats(clock):
    """
    Returns dataframe with US holiday and day-of-week indicators
    :param clock: pandas series of timestamps
    :return: dataframe with holiday and day of week indicators
    """
    df = pd.DataFrame(index=clock.index)
    df['holiday'] = clock.dt.date.astype('datetime64').isin(HOLIDAYS)
    for i in range(6):
        df['dow' + str(i)] = clock.dt.dayofweek == i
    return df


def extract_clock_feats(clock):
    '''
    Creates clock features from timestamps.
    :param clock: pandas series of timestamps.
    :return: pandas dataframe of holiday and day of week indicators, and minute of day.
    '''
    df = extract_day_feats(clock)
    # add in time of day
    sec_norm = (clock.dt.hour * HOUR + clock.dt.minute * MINUTE + clock.dt.second) / DAY
    df['time_of_day'] = np.sin(sec_norm * np.pi)
    df['afternoon'] = sec_norm >= 0.5
    assert all(df.columns == CLOCK_FEATS)
    return df


def get_months_since_lstg(lstg_start, thread_start):
    months = (thread_start - lstg_start) / MONTH
    months = months.rename(MONTHS_SINCE_LSTG)
    assert months.max() < 1
    return months


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
            delay[i] = days[i] / MAX_DELAY['slr']
        elif i in [3, 5]:   # ignore byr arrival and last turn
            delay[i] = days[i] / MAX_DELAY['byr']
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
        if i in IDX['byr']:
            norm[i] = df[i] * (1-norm[i-1]) + (1-df[i]) * norm[i-2]
        elif i in IDX['slr']:
            norm[i] = 1 - df[i] * norm[i-1] - (1-df[i]) * (1-norm[i-2])
    return norm.rename_axis('index', axis=1).stack().astype('float64')


def is_split(con):
    '''
    Boolean for whether concession is (close to) an even split.
    :param con: scalar or Series of concessions.
    :return: boolean or Series of booleans.
    '''
    t = type(con)._name__
    if t == 'float':
        return con in SPLIT_PCTS

    assert 'float' in con.dtype
    if t == 'Series':
        return con.apply(lambda x: x in SPLIT_PCTS)

    if t == 'ndarray':
        return np.apply_along_axis(
            lambda x: x in SPLIT_PCTS, 0, con)

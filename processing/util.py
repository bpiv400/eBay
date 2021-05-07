import numpy as np
import pandas as pd
from agent.const import COMMON_CONS
from utils import extract_clock_feats, byr_norm, slr_norm, unpickle, load_pctile
from constants import PARTS_DIR, START, IDX, DAY, HOLIDAYS, MAX_DELAY_TURN
from featnames import HOLIDAY, DOW_PREFIX, TIME_OF_DAY, AFTERNOON, \
    CLOCK_FEATS, BYR_HIST, SLR, BYR, INDEX


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
        delay[i] = days[i] / MAX_DELAY_TURN
    # no delay larger than 1
    assert delay.max().max() <= 1

    # reshape from wide to long
    days = days.rename_axis('index', axis=1).stack() / DAY
    delay = delay.rename_axis('index', axis=1).stack()

    return days, delay


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
        if i in IDX[BYR]:
            norm[i] = byr_norm(con=df[i],
                               prev_byr_norm=norm[i - 2],
                               prev_slr_norm=norm[i - 1])
        elif i in IDX[SLR]:
            norm[i] = slr_norm(con=df[i],
                               prev_byr_norm=norm[i - 1],
                               prev_slr_norm=norm[i - 2])
    return norm.rename_axis('index', axis=1).stack().astype('float64')


def feat_to_pctile(s, reverse=False, feat=BYR_HIST):
    """
    Converts byr hist counts to percentiles or visa versa.
    :param Series s: counts, or percentiles if reverse.
    :param bool reverse: convert pctile to hist if True.
    :param str feat: feature to convert
    :return: Series
    """
    pc = load_pctile(name=feat)
    if reverse:
        pc = pc.reset_index().set_index('pctile').squeeze()
    v = pc.reindex(index=s.values, method='pad').values
    hist = pd.Series(v, index=s.index, name=feat)
    return hist


def get_lstgs(part):
    """
    Grabs list of partition-specific listing ids.
    :param str part: name of partition
    :return: list of partition-specific listings ids
    """
    d = unpickle(PARTS_DIR + 'partitions.pkl')
    return d[part]


def do_rounding(price):
    """
    Returns booleans for whether offer is round and ends in nines
    :param price: price in dollars and cents
    :return: (bool, bool)
    """
    assert np.min(price) >= .01
    cents = np.round(100 * (price % 1)).astype(np.int8)
    dollars = np.floor(price).astype(np.int64)
    is_nines = (cents >= 90) | np.isin(dollars, [9, 99] + list(range(990, 1000)))
    is_round = (cents == 0) & ~is_nines & \
               (((dollars <= 100) & (dollars % 5 == 0)) | (dollars % 50 == 0))
    assert np.all(~(is_round & is_nines))
    return is_round, is_nines


def get_common_cons(con=None):
    """
    Identifies whether concession is a common concession.
    :param pd.Series con: concessions
    :return: pd.Series
    """
    turn = con.index.get_level_values(INDEX)
    s = pd.Series(False, index=con.index)
    for t in range(1, 7):
        mask = turn == t
        s.loc[mask] = con[mask].apply(lambda x: max(np.isclose(x, COMMON_CONS[t])))
    return s

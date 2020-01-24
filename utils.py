import pickle
import numpy as np
from constants import MAX_DELAY, ARRIVAL_PREFIX, DAY, MONTH, SPLIT_PCTS


def unpickle(file):
    """
    Unpickles a .pkl file encoded with Python 3
    :param file: str giving path to file
    :return: contents of file
    """
    return pickle.load(open(file, "rb"))


def get_remaining(lstg_start, delay_start, max_delay):
    '''
    Calculates number of delay intervals remaining in lstg.
    :param lstg_start: seconds from START to start of lstg.
    :param delay_start: seconds from START to beginning of delay window.
    :param max_delay: length of delay period.
    '''
    remaining = lstg_start + MAX_DELAY[ARRIVAL_PREFIX] - delay_start
    remaining /= max_delay
    remaining = np.minimum(1, remaining)
    return remaining


def extract_clock_feats(seconds):
    '''
    Creates clock features from timestamps.
    :param seconds: seconds since START.
    :return: tuple of time_of_day sine transform and afternoon indicator.
    '''
    sec_norm = (seconds % DAY) / DAY
    time_of_day = np.sin(sec_norm * np.pi)
    afternoon = sec_norm >= 0.5
    return time_of_day, afternoon


def is_split(con):
    '''
    Boolean for whether concession is (close to) an even split.
    :param con: scalar or Series of concessions.
    :return: boolean or Series of booleans.
    '''
    return con in SPLIT_PCTS


def get_months_since_lstg(lstg_start=None, start=None):
    '''
    Float number of months between inputs.
    :param lstg_start: seconds from START to lstg start.
    :param start: seconds from START to focal event.
    :return: number of months between lstg_start and start.
    '''
    return (start - lstg_start) / MONTH


def slr_norm(con=None, prev_byr_norm=None, prev_slr_norm=None):
    '''
    Normalized offer for seller turn.
    :param con: current concession, between 0 and 1.
    :param prev_byr_norm: normalized concession from one turn ago.
    :param prev_slr_norm: normalized concession from two turns ago.
    :return: normalized distance of current offer from start_price to 0.
    '''
    return 1 - con * prev_byr_norm - (1 - prev_slr_norm) * (1 - con)


def byr_norm(con=None, prev_byr_norm=None, prev_slr_norm=None):
    '''
    Normalized offer for buyer turn.
    :param con: current concession, between 0 and 1.
    :param prev_byr_norm: normalized concession from two turns ago.
    :param prev_slr_norm: normalized concession from one turn ago.
    :return: normalized distance of current offer from 0 to start_price.
    '''
    return (1 - prev_slr_norm) * con + prev_byr_norm * (1 - con)
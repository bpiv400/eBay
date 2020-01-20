import pickle, argparse
from compress_pickle import load
import numpy as np, pandas as pd
from constants import MONTH


def get_remaining(lstg_start, delay_start, max_delay):
    remaining = lstg_start + MONTH - delay_start
    remaining /= max_delay
    remaining = np.minimum(1, remaining)
    assert np.all(remaining > 0) and np.all(remaining <= 1)
    return remaining


def unpickle(file):
    """
    Unpickles a .pkl file encoded with Python 3
    :param file: str giving path to file
    :return: contents of file
    """
    return pickle.load(open(file, "rb"))


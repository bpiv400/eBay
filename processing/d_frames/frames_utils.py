import os
from compress_pickle import load
import pandas as pd
from processing.processing_consts import FEATS_DIR


def load_frames(name):
    """
    Loads processed chunk files.
    :param str name: one of ['slr', 'cat', 'cndtn, 'tf_offer']
    :return DataFrame output: concatentated input files.
    """
    # loop and append
    output = []
    n = len([f for f in os.listdir(FEATS_DIR) if name in f])
    for i in range(1,n+1):
        output.append(load(FEATS_DIR + '{}_{}.gz'.format(i, name)))
    output = pd.concat(output).sort_index()
    return output

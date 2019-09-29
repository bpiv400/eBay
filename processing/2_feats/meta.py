import sys, os
sys.path.append('repo/')
sys.path.append('repo/processing/2_feats/')
import argparse
from compress_pickle import dump, load
import numpy as np, pandas as pd
from constants import *
from util import *


if __name__ == "__main__":
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num-1

    # load data
    print('Loading data')
    d = load(CHUNKS_DIR + 'm%d' % num + '.gz')
    L, T, O = [d[k] for k in ['listings', 'threads', 'offers']]

    # categories to strings
    L = categories_to_string(L)

    # set levels for hierarchical time feats
    levels = LEVELS[1:4]

    # create events dataframe
    print('Creating offer events.')
    events = create_events(L, T, O, levels)

    # get upper-level time-valued features
    print('Creating categorical time features') 
    tf = get_cat_time_feats(events, levels)

    # drop flagged lstgs
    print('Restricting observations')
    events = clean_events(events, L)

    # split off listing events
    idx = events.reset_index('thread', drop=True).xs(
        0, level='index').index
    tf = tf.reindex(index=idx)

    # save
    dump(tf, FEATS_DIR + 'm%d' % num + '_tf_meta.gz')

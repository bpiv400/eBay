import sys, os
sys.path.append('repo/')
sys.path.append('repo/processing/2_feats/')
import argparse, pickle
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
    chunk = pickle.load(open(CHUNKS_DIR + 'm%d' % num + '.pkl', 'rb'))
    L, T, O = [chunk[k] for k in ['listings', 'threads', 'offers']]

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

    # save
    pickle.dump(tf, open(FEATS_DIR + 'm%d' % num + '_tf_meta.pkl', 'wb'))

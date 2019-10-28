import sys, os
import argparse
from compress_pickle import dump, load
from datetime import datetime as dt
from constants import *
import processing.b_feats.util as util
import processing.processing_utils as putil


if __name__ == "__main__":
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    parser.add_argument('--feat', action='store', type=int, required=True)
    args = parser.parse_args()
    if args.feat > 7 or args.feat < 1:
        raise RuntimeError("feat must be an integer in range [1, 7]")
    CHUNKS_DIR = 'data/chunks/'
    FEATS_DIR = 'data/feats/'
    # load data
    print('Loading data')
    d = load(CHUNKS_DIR + 'm%d' % args.num + '.gz')
    L, T, O = [d[k] for k in ['listings', 'threads', 'offers']]


    # set levels for hierarchical time feats
    levels = ['meta', 'leaf', 'cndtn']

    # create events dataframe
    print('Creating offer events.')
    start = dt.now()
    events = util.create_events(L, T, O, levels)

    # get upper-level time-valued features
    print('Creating categorical time features') 
    tf = util.get_cat_feats(events, levels, feat_ind=args.feat)
    assert not tf.isna().any().any()
    # save
    end = dt.now()
    print('time: {}'.format((end-start).seconds//3600))
    dump(tf, FEATS_DIR + 'm{}_meta_feats_{}.gz'.format(args.num, args.feat))


    # separate feature sets:


    # lstg counts  - 1
    #   accept counts per listing
    #   slr offers per lstg
    #   byr offers per lstg
    #   threads per lstg
    #   total
    #   open

    # accept norm quantiles - 2

    # concessions - 3
    #   first byr offer quantile
    #   bin probability

    # delay - 4
    #   probability an offer expires (slr)
    #   probability an offer expires (byr)
    #   quantiles

    # start price percentile quantiles - 5

    # history percentile quantiles - 6

    # arrival rate percentile quantiles - 7







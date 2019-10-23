import sys, os
import argparse
from compress_pickle import dump, load
from constants import *
import processing.b_feats.util as util
import processing.processing_utils as putil


if __name__ == "__main__":
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num

    # load data
    print('Loading data')
    d = load(CHUNKS_DIR + 'm%d' % num + '.gz')
    L, T, O = [d[k] for k in ['listings', 'threads', 'offers']]


    # set levels for hierarchical time feats
    levels = LEVELS[1:4]

    # create events dataframe
    print('Creating offer events.')
    events = util.create_events(L, T, O, levels)

    # get upper-level time-valued features
    print('Creating categorical time features') 
    tf = util.get_cat_time_feats(events, levels)

    # save
    dump(tf, FEATS_DIR + 'm%d' % num + '_tf_meta.gz')


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







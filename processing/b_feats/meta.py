import argparse
from compress_pickle import dump, load
from datetime import datetime as dt
import processing.b_feats.util as util


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
    events = load('{}m{}_events.gz'.format(FEATS_DIR, args.num))

    start = dt.now()

    # get upper-level time-valued features
    levels = ['meta', 'leaf', 'cndtn']
    print('Creating categorical time features') 
    tf = util.get_cat_feats(events, levels, feat_ind=args.feat)
    assert not tf.isna().any().any()
    # save
    end = dt.now()
    print('minutes: {}'.format((end-start).seconds//60))
    dump(tf, FEATS_DIR + 'm{}_cat_feats_{}.gz'.format(args.num, args.feat))



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







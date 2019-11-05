import argparse
from compress_pickle import dump, load
from datetime import datetime as dt
import processing.b_feats.util as util
from constants import FEATS_DIR, CHUNKS_DIR

if __name__ == "__main__":
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    args = parser.parse_args()

    # load data
    print('Loading data...')
    d = load(CHUNKS_DIR + 'm%d' % args.num + '.gz')
    L, T, O = [d[k] for k in ['listings', 'threads', 'offers']]

    # set levels for hierarchical time feats
    levels = ['category', 'cndtn']

    # create events dataframe
    print('Creating offer events...')
    events = util.create_events(L, T, O, levels)

    print('Creating categorical time features...')
    start = dt.now()
    tf = util.get_all_cat_feats(events, levels)
    assert not tf.isna().any().any()
    # save
    end = dt.now()
    print('minutes: {}'.format((end-start).seconds//60))
    dump(tf, FEATS_DIR + 'm{}_cat_feats_{}.gz'.format(args.num, args.feat))

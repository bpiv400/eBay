import argparse
from compress_pickle import dump, load
from datetime import datetime as dt
import processing.c_feats.util as util
from constants import FEATS_DIR, CHUNKS_DIR


def main():
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    parser.add_argument('--slr', action='store_true', default=False)
    args = parser.parse_args()

    if not args.slr:
        print('')
        chunk_name = 'cat'
        levels = ['cat', 'cndtn']
    else:
        print('making slr features')
        chunk_name = 'slr'
        levels = ['slr']

    # load data
    print('Loading data...')
    d = load('{}{}{}.gz'.format(CHUNKS_DIR, chunk_name, args.num))
    L, T, O = [d[k] for k in ['listings', 'threads', 'offers']]

    # set levels for hierarchical time feats
    print('Creating offer events...')
    events = util.create_events(L, T, O, levels)

    print('Creating categorical time features...')
    start = dt.now()
    tf = util.get_all_cat_feats(events, levels)
    assert not tf.isna().any().any()
    end = dt.now()
    print('minutes: {}'.format((end - start).seconds // 60))

    print('Saving...')
    dump(tf, '{}{}_{}.gz'.format(FEATS_DIR, args.num, chunk_name))


if __name__ == "__main__":
    main()

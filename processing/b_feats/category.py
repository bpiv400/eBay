import argparse
from compress_pickle import dump, load
from datetime import datetime as dt
import processing.b_feats.util as util
from constants import FEATS_DIR, CHUNKS_DIR, LEVELS


def get_multi_lstgs(L):
    df = L[LEVELS[:-1] + ['start_date', 'end_time']].set_index(
        LEVELS[:-1], append=True).reorder_levels(LEVELS).sort_index()
    # start time
    df['start_date'] *= 24 * 3600
    df = df.rename(lambda x: x.split('_')[0], axis=1)
    # find multi-listings
    df = df.sort_values(df.index.names[:-1] + ['start'])
    maxend = df.end.groupby(df.index.names[:-1]).cummax()
    maxend = maxend.groupby(df.index.names[:-1]).shift(1)
    overlap = df.start <= maxend
    return overlap.groupby(df.index.names).max()


def clean_events(events, L):
    # identify multi-listings
    ismulti = get_multi_lstgs(L)
    ismulti.index = ismulti.index.droplevel(level=['cat', 'cndtn'])
    # drop multi-listings
    events = events[~ismulti.reindex(index=events.index)]
    # limit index to ['lstg', 'thread', 'index']
    events = events.reset_index('slr', drop=True).sort_index()
    # 30-day burn in
    events = events.join(L['start_date'])
    events = events[events.start_date >= 30].drop('start_date', axis=1)
    # drop listings in which prices have changed
    events = events[events.flag == 0].drop('flag', axis=1)
    return events


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
        chunk_name = ''
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

    # clean slr features
    if args.slr:
        # drop flagged lstgs
        print('Restricting observations...')
        events = clean_events(events, L)

        # split off listing events
        idx = events.reset_index('thread', drop=True).xs(
            0, level='index').index
        tf = tf.reindex(index=idx)
        events = events.drop(0, level='thread')
        # save events
        dump(events, '{}{}_events.gz'.format(FEATS_DIR, args.num))
        chunk_name = 'slr'

    print('Saving...')
    dump(tf, '{}{}_feat_{}.gz'.format(FEATS_DIR, chunk_name, args.num))


if __name__ == "__main__":
    main()

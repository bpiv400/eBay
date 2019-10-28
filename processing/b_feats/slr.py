import sys
import os
import argparse
from compress_pickle import dump, load
from processing.b_feats.util import get_cat_feats, create_events
from constants import CHUNKS_DIR, FEATS_DIR, LEVELS


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
    ismulti.index = ismulti.index.droplevel(level=['meta', 'leaf', 'cndtn'])
    # drop multi-listings
    print(events.index.names)
    print(ismulti.index.names)
    events = events[~ismulti.reindex(index=events.index)]
    # limit index to ['lstg', 'thread', 'index']
    events = events.reset_index('slr', drop=True).sort_index()
    # 30-day burn in
    events = events.join(L['start_date'])
    events = events[events.start_date >= 30].drop('start_date', axis=1)
    # drop listings in which prices have changed
    events = events[events.flag == 0].drop('flag', axis=1)
    return events


if __name__ == "__main__":
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    parser.add_argument('--feat', action='store', type=int, required=True)
    args = parser.parse_args()


    # load data
    print('Loading data')
    d = load(CHUNKS_DIR + '%d' % args.num + '.gz')
    L, T, O = [d[k] for k in ['listings', 'threads', 'offers']]

    # set levels for hierarchical time feats
    levels = ['slr']

    # create events dataframe
    print('Creating offer events.')
    events = create_events(L, T, O, levels)

    # get upper-level time-valued features
    print('Creating hierarchical time features') 
    tf_slr = get_cat_feats(events, levels, feat_ind=args.feat)

    # drop flagged lstgs
    print('Restricting observations')
    events = clean_events(events, L)

    # split off listing events
    idx = events.reset_index('thread', drop=True).xs(
        0, level='index').index
    tf_slr = tf_slr.reindex(index=idx)
    events = events.drop(0, level='thread') # remove lstg start/end obs

    tf_file = '{}{}_slr_feat_{}.gz'.format(FEATS_DIR, args.num, args.feat)
    dump(tf_slr, tf_file)
    if args.feat == 7:
        events_file = '{}{}_events.gz'.format(FEATS_DIR, args.num)
        dump(events, events_file)
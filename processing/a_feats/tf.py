import numpy as np
import pandas as pd
from datetime import datetime as dt
from processing.a_feats.util import collapse_dict
from utils import unpickle, topickle
from processing.util import run_func_on_chunks
from constants import START, IDX, FEATS_DIR
from featnames import SLR, BYR, LSTG, THREAD, INDEX, ACCEPT, REJECT, CLOCK, NORM


def open_offers(df, levels, role):
    # index number
    if INDEX in df.columns:
        index = df[INDEX]
    else:
        index = df.index.get_level_values(INDEX)
    # open and closed markers
    if role == SLR:
        start = ~df[BYR] & ~df[ACCEPT] & (index > 0)
        end = df.byr & (index > 1)
    else:
        start = df[BYR] & ~df[REJECT] & ~df[ACCEPT]
        end = ~df[BYR] & (index > 1)
    # open - closed
    s = start.astype(np.int64) - end.astype(np.int64)
    # cumulative sum by levels grouping
    return s.groupby(by=levels).cumsum()


def thread_count(subset, full=False):
    df = subset.copy()
    index_names = [ind for ind in df.index.names if ind != INDEX]
    thread_counter = df.reset_index()[INDEX] == 1
    thread_counter.index = df.index
    total_threads = thread_counter.reset_index(THREAD).thread.groupby(LSTG).max()
    thread_counter = thread_counter.unstack(THREAD)
    total_threads = total_threads.reindex(index=thread_counter.index,
                                          level=LSTG)
    if full:
        counts = thread_counter.sum(axis=1)
        counts = counts.groupby('lstg').cumsum()
        counts = conform_cut(counts)
        return counts
    else:
        count = {}
        # count and max over non-focal threads
        for n in thread_counter.columns:
            # restrict observations
            cut = thread_counter.drop(n, axis=1).loc[total_threads >= n]
            counts = cut.sum(axis=1)
            counts = counts.groupby('lstg').cumsum()
            count[n] = conform_cut(counts)
        # concat into series and return
        return collapse_dict(count, index_names)


def conform_cut(cut):
    cut = cut.reset_index('index', drop=True)
    cut = cut.groupby(level=cut.index.names).last()
    return cut


def rolling_feat(count, sum_feat=False):
    count = count.reset_index('clock', drop=False)
    count = count.groupby('lstg')
    if sum_feat:
        count = count.apply(lambda x: x.rolling('172800s', on='clock', min_periods=1).sum())
    else:
        count = count.apply(lambda x: x.rolling('172800s', on='clock', min_periods=1).max())
    count = count.set_index('clock', append=True, drop=True).squeeze()
    return count


def recent_count(offer_counter):
    # sum across threads
    count = offer_counter.sum(axis=1)
    count = rolling_feat(count, sum_feat=True)
    return count


def full_recent(max_offers=None, offer_counter=None):
    count = recent_count(offer_counter)
    best = recent_best(max_offers)
    return conform_cut(count), conform_cut(best)


def prepare_counters(df, role):
    if role == 'byr':
        offer_counter = df[BYR] & ~df[REJECT]
    else:
        offer_counter = ~df[BYR] & ~df[REJECT]
    thread_counter = offer_counter.reset_index(THREAD).thread.groupby(LSTG).max()
    offer_counter = offer_counter.unstack(level=THREAD)
    thread_counter = thread_counter.reindex(index=offer_counter.index,
                                            level=LSTG)
    return offer_counter, thread_counter


def recent_best(max_cut):
    curr_best = max_cut.max(axis=1).fillna(0.0)
    curr_best = rolling_feat(curr_best, sum_feat=False)
    return curr_best


def exclude_focal_recent(max_offers=None, offer_counter=None, thread_counter=None):
    # initialize dictionaries for later concatenation
    count = {}
    best = {}
    for n in max_offers.columns:
        # restrict observations
        max_cut = max_offers.drop(n, axis=1).loc[thread_counter >= n]
        counter_cut = offer_counter.drop(n, axis=1).loc[thread_counter >= n]
        # sum across threads
        count[n] = recent_count(counter_cut)
        # best offer
        curr_best = recent_best(max_cut)
        # re-conform
        best[n] = conform_cut(curr_best)
        count[n] = conform_cut(count[n])
    return count, best


def get_recent_feats(subset, role, full=False):
    df, index_names = prepare_feat(subset)
    if role == SLR:
        df.loc[df.byr, NORM] = np.nan
    elif role == BYR:
        df.loc[~df[BYR], NORM] = np.nan
    df.loc[df[REJECT], NORM] = np.nan
    converter = df.index.levels[df.index.names.index(CLOCK)]
    converter = pd.to_datetime(converter, origin=START, unit='s')
    df.index = df.index.set_levels(converter, level=CLOCK)
    # use max value for thread events at same clock time
    max_offers = df.norm.groupby(df.index.names).max()
    max_offers = max_offers.unstack(level=THREAD)

    offer_counter, thread_counter = prepare_counters(df, role)

    if not full:
        count, best = exclude_focal_recent(max_offers=max_offers,
                                           offer_counter=offer_counter,
                                           thread_counter=thread_counter)
        # concat into series
        count = collapse_dict(count, index_names)
        best = collapse_dict(best, index_names)
    else:
        count, best = full_recent(max_offers=max_offers,
                                  offer_counter=offer_counter)

    count = fix_clock(count)
    best = fix_clock(best)
    return count, best


def fix_clock(df):
    start = pd.to_datetime(START)
    diff = df.index.levels[df.index.names.index(CLOCK)] - start
    diff = diff.total_seconds().astype(int)
    df.index = df.index.set_levels(diff, level=CLOCK)
    return df


def exclude_focal(max_offers=None, offer_counter=None, thread_counter=None,
                  role=False, is_open=False, open_counter=None):
    count = {}
    best = {}
    # count and max over non-focal threads
    for n in max_offers.columns:
        # restrict observations
        max_cut = max_offers.drop(n, axis=1).loc[thread_counter >= n]
        counter_cut = offer_counter.drop(n, axis=1).loc[thread_counter >= n]
        if is_open and role == 'slr':
            open_cut = open_counter.drop(n, axis=1).loc[thread_counter >= n]
            count[n] = open_cut.sum(axis=1)
        elif is_open:
            count[n] = (max_cut > 0).sum(axis=1)
        else:
            counts = counter_cut.sum(axis=1)
            count[n] = counts.groupby(LSTG).cumsum()

        # best offer
        curr_best = max_cut.max(axis=1).fillna(0.0)
        best[n] = conform_cut(curr_best)
        count[n] = conform_cut(count[n])
    return count, best


def full_feat(max_offers=None, offer_counter=None, role=False, is_open=False, open_counter=None):
    # count and max over all threads
    if is_open and role == SLR:
        count = open_counter.sum(axis=1)
    elif is_open:
        count = (max_offers > 0).sum(axis=1)
    else:
        count = offer_counter.sum(axis=1)
        count = count.groupby(LSTG).cumsum()
    # best offer
    best = max_offers.max(axis=1).fillna(0.0)
    # keep only later timestep
    count = conform_cut(count)
    best = conform_cut(best)
    return count, best


def prepare_feat(subset):
    df = subset.copy()
    df[BYR] = df[BYR].astype(bool)
    df[REJECT] = df[REJECT].astype(bool)
    df[ACCEPT] = df[ACCEPT].astype(bool)
    index_names = [ind for ind in df.index.names if ind != 'index']
    return df, index_names


def add_lstg_time_feats(subset, role, is_open, full=False):
    df, index_names = prepare_feat(subset)
    if is_open:
        open_counter = open_offers(df, [LSTG, THREAD], role)
        assert (open_counter.max() <= 1) & (open_counter.min() == 0)
        df.loc[open_counter == 0, NORM] = 0.0
        if role == SLR:
            open_counter = open_counter.unstack(level=THREAD).groupby(LSTG).ffill()
    else:
        open_counter = None
        if role == SLR:
            df.loc[df[BYR], NORM] = np.nan
        elif role == BYR:
            df.loc[~df[BYR], NORM] = np.nan
    # use max value for thread events at same clock time
    max_offers = df.norm.groupby(df.index.names).max()
    # unstack by thread and fill with last value
    max_offers = max_offers.unstack(level=THREAD).groupby(LSTG).ffill()
    offer_counter, thread_counter = prepare_counters(df, role)
    # initialize dictionaries for later concatenation
    if not full:
        count, best = exclude_focal(max_offers=max_offers,
                                    offer_counter=offer_counter,
                                    thread_counter=thread_counter,
                                    open_counter=open_counter,
                                    role=role, is_open=is_open)
        # concat into series and return
        return collapse_dict(count, index_names), collapse_dict(best, index_names)
    else:
        return full_feat(max_offers=max_offers,
                         offer_counter=offer_counter,
                         open_counter=open_counter,
                         role=role,
                         is_open=is_open)


def get_lstg_time_feats(events, full=False):
    # create dataframe for variable creation
    ordered = events.sort_values([LSTG, CLOCK, INDEX]).drop(
        ['message', 'price'], axis=1)
    # identify listings with multiple threads
    if not full:
        threads = ordered.reset_index().groupby(LSTG)[THREAD].nunique()
        check = threads > 1
        subset = ordered.loc[check[check].index].set_index([CLOCK], append=True)
    else:
        subset = ordered.set_index([CLOCK], append=True)
    subset = subset.reorder_levels([LSTG, THREAD, CLOCK, INDEX])
    # add features for open offers
    tf = pd.DataFrame()
    for role in [SLR, BYR]:
        cols = [role + c for c in ['_offers', '_best']]
        for is_open in [False, True]:
            if is_open:
                cols = [c + '_open' for c in cols]
            tf[cols[0]], tf[cols[1]] = add_lstg_time_feats(
                subset, role, is_open, full=full)
        cols = [role + c for c in ['_offers_recent', '_best_recent']]
        tf[cols[0]], tf[cols[1]] = get_recent_feats(subset, role, full=full)
    tf['thread_count'] = thread_count(subset, full=full)
    # error checking
    assert np.min(tf.byr_offers >= tf.slr_offers)
    assert np.min(tf.byr_offers >= tf.byr_offers_open)
    assert np.min(tf.slr_best >= tf.slr_best_open)
    assert np.min(tf.byr_best >= tf.byr_best_open)
    # sort and return
    return tf


def get_diffs(grouped_df, df, remove_zeros=True):
    diff = grouped_df.diff()
    # find the first row in each lstg and replace with raw features
    firsts = diff.isna().any(axis=1)
    print(firsts.index.names)
    diff.loc[firsts, :] = df.loc[firsts, :]
    if remove_zeros:
        diff = drop_zeros(diff)
    return diff


def drop_zeros(diff):
    # subset to only time steps where a change occurs
    nonzeros = (diff != 0).any(axis=1)
    diff = diff.loc[nonzeros, :]
    return diff


def arrival_time_feats(tf_lstg):
    df = tf_lstg.copy()
    # sort and group
    df = df.sort_index(level=CLOCK)
    diff = get_diffs(df.groupby(LSTG), df)
    return diff


def prepare_index_join(tf, events):
    events = events.reset_index(drop=False)
    # one entry for each lstg, thread, index
    events = events[[LSTG, THREAD, CLOCK, INDEX]]
    events = events.set_index([LSTG, THREAD, CLOCK],
                              append=False, drop=True)
    events = events.reorder_levels(tf.index.names)
    # subset events to lstgs contained in tf
    event_lstgs = events.index.get_level_values(LSTG)
    subset_lstgs = tf.index.get_level_values(LSTG)
    events = events.loc[event_lstgs.isin(subset_lstgs), :]
    return events


def subset_to_turns(tf, events):
    events = prepare_index_join(tf, events)
    # subset tf to only include turns
    # add index to tf
    tf = tf.join(events, how='inner')
    tf = tf.set_index(INDEX, append=True, drop=True)
    # drop clock
    tf = tf.reset_index(CLOCK, drop=True)
    tf = tf.sort_index(level=[LSTG, THREAD, INDEX])
    return tf


def con_time_feats(tf_lstg, events):
    tf = subset_to_turns(tf_lstg.copy(), events.copy())
    print('getting diffs')
    diff = get_diffs(tf.groupby([LSTG, THREAD]), tf, remove_zeros=True)
    return diff


def delay_time_feats(tf_lstg, events):
    tf_lstg = tf_lstg.copy()
    # compute timestep differences
    deltas = get_diffs(tf_lstg.groupby([LSTG, THREAD]), tf_lstg, remove_zeros=False)
    deltas = add_deltas_index(deltas, events.copy())
    deltas = drop_zeros(deltas)
    return deltas


def add_deltas_index(deltas, events):
    # prepare events
    events = prepare_index_join(deltas, events)
    events = events.rename(columns={INDEX: 'ind'})
    events = events.sort_values(by='ind')
    events = events.groupby(by=[LSTG, THREAD, CLOCK]).first()

    # add index to deltas
    deltas = deltas.join(events, how='left')
    deltas = deltas.reset_index(drop=False)
    deltas = deltas.sort_values([LSTG, THREAD, CLOCK, 'ind'])
    deltas = deltas.reset_index(drop=False)
    deltas = deltas.set_index([LSTG, THREAD], drop=True, append=False)
    deltas = deltas.groupby(level=[LSTG, THREAD]).bfill()
    deltas = deltas.set_index(['ind', CLOCK], drop=True, append=True)

    # drop first turn and nans
    # nans occur because some rows give values after the last turn in a thread
    not_first = deltas.index.get_level_values('ind') != 1
    not_nan = ~deltas.index.get_level_values('ind').isna()
    drops = (not_first.astype(bool) & not_nan.astype(bool)).astype(bool)
    deltas = deltas.loc[drops, :]

    # rename index and check for nans
    deltas.index = deltas.index.rename(names=INDEX, level='ind')
    assert (~deltas.index.get_level_values(INDEX).isna()).all()
    return deltas


def create_tf(chunk=None):
    start = dt.now()
    events = unpickle(FEATS_DIR + 'chunks/{}.pkl'.format(chunk))['offers']
    print('{} offers'.format(len(events)))
    events[BYR] = events.index.isin(IDX[BYR], level=INDEX)
    tf_lstg_focal = get_lstg_time_feats(events, full=False)
    con_feats = con_time_feats(tf_lstg_focal, events)
    assert not con_feats.isna().any().any()
    print('{} seconds'.format((dt.now() - start).total_seconds()))
    return con_feats


def main():
    res = run_func_on_chunks(f=create_tf, func_kwargs=dict())
    df = pd.concat(res).sort_index()
    topickle(df, FEATS_DIR + 'tf.pkl')


if __name__ == "__main__":
    main()

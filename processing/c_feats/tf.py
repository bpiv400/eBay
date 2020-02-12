import argparse
from compress_pickle import dump, load
from processing.c_feats.util import *
from processing.processing_utils import get_con, get_norm
from constants import *
from processing.processing_consts import *


def thread_count(subset, full=False):
    df = subset.copy()
    index_names = [ind for ind in df.index.names if ind != 'index']
    thread_counter = df.reset_index()['index'] == 1
    thread_counter.index = df.index
    total_threads = thread_counter.reset_index('thread').thread.groupby('lstg').max()
    thread_counter = thread_counter.unstack('thread')
    total_threads = total_threads.reindex(index=thread_counter.index,
                                          level='lstg')
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
        offer_counter = df.byr & ~df.reject & ~df.censored
    else:
        offer_counter = ~df.byr & ~df.reject & ~df.censored
    thread_counter = offer_counter.reset_index('thread').thread.groupby('lstg').max()
    offer_counter = offer_counter.unstack(level='thread')
    thread_counter = thread_counter.reindex(index=offer_counter.index, level='lstg')
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
    if role == 'slr':
        df.loc[df.byr, 'norm'] = np.nan
    elif role == 'byr':
        df.loc[~df.byr, 'norm'] = np.nan
    df.loc[df.reject, 'norm'] = np.nan
    converter = df.index.levels[df.index.names.index('clock')]
    converter = pd.to_datetime(converter, origin=START, unit='s')
    df.index = df.index.set_levels(converter, level='clock')
    # use max value for thread events at same clock time
    max_offers = df.norm.groupby(df.index.names).max()
    max_offers = max_offers.unstack(level='thread')

    offer_counter, thread_counter = prepare_counters(df, role)

    if not full:
        count, best = exclude_focal_recent(max_offers=max_offers, offer_counter=offer_counter,
                                           thread_counter=thread_counter)
        # concat into series
        count = collapse_dict(count, index_names)
        best = collapse_dict(best, index_names)
    else:
        count, best = full_recent(max_offers=max_offers, offer_counter=offer_counter)

    count = fix_clock(count)
    best = fix_clock(best)
    return count, best


def fix_clock(df):
    start = pd.to_datetime(START)
    diff = df.index.levels[df.index.names.index('clock')] - start
    diff = diff.total_seconds().astype(int)
    df.index = df.index.set_levels(diff, level='clock')
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
            count[n] = counts.groupby('lstg').cumsum()

        # best offer
        curr_best = max_cut.max(axis=1).fillna(0.0)
        best[n] = conform_cut(curr_best)
        count[n] = conform_cut(count[n])
    return count, best


def full_feat(max_offers=None, offer_counter=None, role=False, is_open=False, open_counter=None):
    # count and max over all threads
    if is_open and role == 'slr':
        count = open_counter.sum(axis=1)
    elif is_open:
        count = (max_offers > 0).sum(axis=1)
    else:
        count = offer_counter.sum(axis=1)
        count = count.groupby('lstg').cumsum()

    # best offer
    best = max_offers.max(axis=1).fillna(0.0)

    # keep only later timestep
    count = conform_cut(count)
    best = conform_cut(best)
    return count, best


def prepare_feat(subset):
    df = subset.copy()
    df.byr = df.byr.astype(bool)
    df.reject = df.reject.astype(bool)
    df.accept = df.accept.astype(bool)
    df.censored = df.censored.astype(bool)
    index_names = [ind for ind in df.index.names if ind != 'index']
    return df, index_names


def add_lstg_time_feats(subset, role, is_open, full=False):
    df, index_names = prepare_feat(subset)
    if is_open:
        open_counter = open_offers(df, ['lstg', 'thread'], role)
        assert (open_counter.max() <= 1) & (open_counter.min() == 0)
        df.loc[open_counter == 0, 'norm'] = 0.0
        if role == 'slr':
            open_counter = open_counter.unstack(level='thread').groupby('lstg').ffill()
    else:
        open_counter = None
        if role == 'slr':
            df.loc[df.byr, 'norm'] = np.nan
        elif role == 'byr':
            df.loc[~df.byr, 'norm'] = np.nan
    df.loc[df.censored, 'norm'] = np.nan
    # use max value for thread events at same clock time
    max_offers = df.norm.groupby(df.index.names).max()
    # unstack by thread and fill with last value
    max_offers = max_offers.unstack(level='thread').groupby('lstg').ffill()
    offer_counter, thread_counter = prepare_counters(df, role)
    # initialize dictionaries for later concatenation
    if not full:
        count, best = exclude_focal(max_offers=max_offers, offer_counter=offer_counter,
                                    thread_counter=thread_counter, open_counter=open_counter,
                                    role=role, is_open=is_open)
        # concat into series and return
        return collapse_dict(count, index_names), collapse_dict(best, index_names)
    else:
        return full_feat(max_offers=max_offers, offer_counter=offer_counter,
                         open_counter=open_counter, role=role, is_open=is_open)


def get_lstg_time_feats(events, full=False):
    # create dataframe for variable creation
    ordered = events.sort_values(['lstg', 'clock', 'index', 'censored']).drop(
        ['message', 'price'], axis=1)
    # identify listings with multiple threads
    if not full:
        threads = ordered.reset_index().groupby('lstg')['thread'].nunique()
        check = threads > 1
        subset = ordered.loc[check[check].index].set_index(['clock'], append=True)
    else:
        subset = ordered.set_index(['clock'], append=True)
    subset = subset.reorder_levels(['lstg', 'thread', 'clock', 'index'])
    # add features for open offers
    tf = pd.DataFrame()
    for role in ['slr', 'byr']:
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
    assert (tf.byr_offers >= tf.slr_offers).min()
    assert (tf.byr_offers >= tf.byr_offers_open).min()
    assert (tf.slr_best >= tf.slr_best_open).min()
    assert (tf.byr_best >= tf.byr_best_open).min()
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

def arrival_time_feats(tf_lstg, events):
    df = tf_lstg.copy()
    # sort and group
    df = df.sort_index(level='clock')
    diff = get_diffs(df.groupby('lstg'), df)
    return diff


def prepare_index_join(tf, events):
    events = events.reset_index(drop=False)
    # one entry for each lstg, thread, index
    events = events[['lstg', 'thread', 'clock', 'index']]
    events = events.set_index(['lstg', 'thread', 'clock'], append=False, drop=True)
    events = events.reorder_levels(tf.index.names)
    # subset events to lstgs contained in tf
    event_lstgs = events.index.get_level_values('lstg')
    subset_lstgs = tf.index.get_level_values('lstg')
    events = events.loc[event_lstgs.isin(subset_lstgs), :]
    return events


def subset_to_turns(tf, events):
    events = prepare_index_join(tf, events)
    # subset tf to only include turns
    # add index to tf
    tf = tf.join(events, how='inner')
    tf = tf.set_index('index', append=True, drop=True)
    # drop clock
    tf = tf.reset_index('clock', drop=True)
    tf = tf.sort_index(level=['lstg', 'thread', 'index'])
    return tf


def con_time_feats(tf_lstg, events):
    tf = subset_to_turns(tf_lstg.copy(), events.copy())
    print('getting diffs')
    diff = get_diffs(tf.groupby(['lstg', 'thread']), tf, remove_zeros=True)
    return diff


def delay_time_feats(tf_lstg, events):
    tf_lstg = tf_lstg.copy()
    # compute timestep differences
    deltas = get_diffs(tf_lstg.groupby(['lstg', 'thread']), tf_lstg, remove_zeros=False)
    deltas = add_deltas_index(deltas, events.copy())
    deltas = drop_zeros(deltas)
    return deltas


def add_deltas_index(deltas, events):
    # prepare events
    events = prepare_index_join(deltas, events)
    events = events.rename(columns={'index': 'ind'})
    events = events.sort_values(by='ind')
    events = events.groupby(by=['lstg', 'thread', 'clock']).first()

    # add index to deltas
    deltas = deltas.join(events, how='left')
    deltas = deltas.reset_index(drop=False)
    deltas = deltas.sort_values(['lstg', 'thread', 'clock', 'ind'])
    deltas = deltas.reset_index(drop=False)
    deltas = deltas.set_index(['lstg', 'thread'], drop=True, append=False)
    deltas = deltas.groupby(level=['lstg', 'thread']).bfill()
    deltas = deltas.set_index(['ind', 'clock'], drop=True, append=True)

    # drop first turn and nans
    # nans occur because some rows give values after the last turn in a thread
    not_first = deltas.index.get_level_values('ind') != 1
    not_nan = ~deltas.index.get_level_values('ind').isna()
    drops = (not_first.astype(bool) & not_nan.astype(bool)).astype(bool)
    deltas = deltas.loc[drops, :]

    # rename index and check for nans
    deltas.index = deltas.index.rename(names='index', level='ind')
    assert (~deltas.index.get_level_values('index').isna()).all()
    return deltas


def output_path(model, num):
    return FEATS_DIR + '{}_tf_{}.gz'.format(num, model)


def prepare_events(num):
    d = load('{}slr{}.gz'.format(CHUNKS_DIR, num))
    L, events = [d[k] for k in ['listings', 'offers']]

    # drop flagged listings
    events = events.join(L.flag).sort_index()
    events = events.loc[~events.flag, :]
    events = events.drop('flag', axis=1)

    # buyer turn indicator
    events['byr'] = events.index.isin(IDX['byr'], level='index')

    # add normalized offer to events
    con = get_con(events.price.unstack(), L.start_price)
    events['norm'] = get_norm(con)
    return events


def main():
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    parser.add_argument('--arrival', action='store_true', default=False)
    args = parser.parse_args()
    num, arrival = args.num, args.arrival

    # load data
    print('Loading data: %d' % num)
    events = prepare_events(args.num)

    # create lstg-level time-valued features
    print('Creating lstg-level time-valued features')
    if not arrival:
        tf_lstg_focal = get_lstg_time_feats(events, full=False)

        print('preparing concession output...')
        con_feats = con_time_feats(tf_lstg_focal, events)
        assert not con_feats.isna().any().any()
        dump(con_feats, output_path('offer', num))

        # print('preparing delay output...')
        # delay_feats = delay_time_feats(tf_lstg_focal, events)
        # assert not delay_feats.isna().any().any()
        # dump(delay_feats, output_path('delay_diff', num))
        
    else:
        tf_lstg_full = get_lstg_time_feats(events, full=True)
        print('preparing arrival output...')
        arrival_feats = arrival_time_feats(tf_lstg_full)
        assert not arrival_feats.isna().any().any()
        dump(arrival_feats, output_path('arrival', num))


if __name__ == "__main__":
    main()

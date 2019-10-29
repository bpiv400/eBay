import argparse
from compress_pickle import dump, load
from processing.b_feats.util import *
from constants import *


def thread_count(subset, full=False):
    df = subset.copy()
    index_names = [ind for ind in df.index.names if ind != 'index']
    thread_counter = df.reset_index()['index'] == 1
    thread_counter.index = df.index
    total_threads = thread_counter.reset_index('thread').thread.groupby('lstg').max()
    thread_counter = thread_counter.unstack('thread')
    total_threads = total_threads.reindex(index=thread_counter.index, level='lstg')
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
    print('full recent count')
    count = recent_count(offer_counter)
    print('full recent best')
    best = recent_best(max_offers)
    print('done best')
    return conform_cut(count), conform_cut(best)


def prepare_counters(df, role):
    if role == 'byr':
        offer_counter = df.byr & ~df.reject
    else:
        offer_counter = ~df.byr & ~df.reject
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
        print('full recent')
        count, best = full_recent(max_offers=max_offers, offer_counter=offer_counter)

    print('count clock...')
    count = fix_clock(count)
    print('best clock...')
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
    index_names = [ind for ind in df.index.names if ind != 'index']
    return df, index_names


def add_lstg_time_feats(subset, role, is_open, full=False):
    df, index_names = prepare_feat(subset)
    if is_open:
        open_counter = open_offers(df, ['lstg', 'thread'], role)
        assert (open_counter.max() == 1) & (open_counter.min() == 0)
        df.loc[open_counter == 0, 'norm'] = 0.0
        if role == 'slr':
            open_counter = open_counter.unstack(level='thread').groupby('lstg').ffill()
    else:
        open_counter = None
        if role == 'slr':
            df.loc[df.byr, 'norm'] = np.nan
        elif role == 'byr':
            df.loc[~df.byr, 'norm'] = np.nan
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
            print('role: {}, open: {}'.format(role, is_open))
            if is_open:
                cols = [c + '_open' for c in cols]
            tf[cols[0]], tf[cols[1]] = add_lstg_time_feats(
                subset, role, is_open, full=full)
        cols = [role + c for c in ['_offers_recent', '_best_recent']]
        print('recent')
        tf[cols[0]], tf[cols[1]] = get_recent_feats(subset, role, full=full)
    print('thread count')
    tf['thread_count'] = thread_count(subset, full=full)
    # error checking
    assert (tf.byr_offers >= tf.slr_offers).min()
    assert (tf.byr_offers >= tf.byr_offers_open).min()
    assert (tf.slr_best >= tf.slr_best_open).min()
    assert (tf.byr_best >= tf.byr_best_open).min()
    # sort and return
    return tf


if __name__ == "__main__":
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num

    # load data
    print('Loading data')
    FEATS_DIR = 'data/feats/'
    print(FEATS_DIR + '%d_events.gz' % num)
    events = load(FEATS_DIR + '%d_events.gz' % num)

    # create lstg-level time-valued features
    print('Creating lstg-level time-valued features')
    events['norm'] = events.price / events.start_price
    events.loc[~events['byr'], 'norm'] = 1 - events['norm']
    tf_lstg_focal = get_lstg_time_feats(events, full=False)
    # tf_lstg_full = get_lstg_time_feats(events, full=True)

    # save
    dump(tf_lstg_focal, FEATS_DIR + '%d_tf_lstg.gz' % num)

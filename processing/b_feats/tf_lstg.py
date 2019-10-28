import argparse
from compress_pickle import dump, load
from processing.b_feats.util import *
from constants import *


def thread_count(subset):
    df = subset.copy()
    index_names = [ind for ind in df.index.names if ind != 'index']
    thread_counter = df.reset_index()['index'] == 1
    thread_counter.index = df.index
    total_threads = thread_counter.reset_index('thread').thread.groupby('lstg').max()
    thread_counter = thread_counter.unstack('thread')
    total_threads = total_threads.reindex(index=thread_counter.index, level='lstg')
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


def recent_count(subset, role):
    df = subset.copy()
    df.byr = df.byr.astype(bool)
    df.reject = df.reject.astype(bool)
    df.accept = df.accept.astype(bool)
    index_names = [ind for ind in df.index.names if ind != 'index']
    if role == 'slr':
        df.loc[df.byr, 'norm'] = np.nan
    elif role == 'byr':
        df.loc[~df.byr, 'norm'] = np.nan
    df.index = df.set_levels(pd.to_datetime(df.index.get_level_values('clock'),
                                            origin=START, unit='s'),
                             level='clock')
    # use max value for thread events at same clock time
    max_offers = df.norm.groupby(df.index.names).max()
    max_offers = max_offers.unstack(level='thread')

    if role == 'byr':
        offer_counter = df.byr & ~df.reject
    else:
        offer_counter = ~df.byr & ~df.reject
    thread_counter = offer_counter.reset_index('thread').thread.groupby('lstg').max()
    offer_counter = offer_counter.unstack(level='thread')

    thread_counter = thread_counter.reindex(index=offer_counter.index, level='lstg')
    # initialize dictionaries for later concatenation
    count = {}
    best = {}
    for n in max_offers.columns:
        # restrict observations
        max_cut = max_offers.drop(n, axis=1).loc[thread_counter >= n]
        counter_cut = offer_counter.drop(n, axis=1).loc[thread_counter >= n]

        # sum across threads
        counts = counter_cut.sum(axis=1)
        counts = counts.groupby('lstg')
        count[n] = counts.apply(lambda x: x.rolling('172800s', on='clock').cumsum())

        # best offer
        curr_best = max_cut.max(axis=1).fillna(0.0)
        curr_best = curr_best.groupby('lstg')
        curr_best = curr_best.apply(lambda x: x.rolling('172800s', on='clock').cummax())

        # reconform
        best[n] = conform_cut(curr_best)
        count[n] = conform_cut(count[n])
    # concat into series and return
    return collapse_dict(count, index_names), collapse_dict(best, index_names)


def add_lstg_time_feats(subset, role, is_open):
    df = subset.copy()
    df.byr = df.byr.astype(bool)
    df.reject = df.reject.astype(bool)
    df.accept = df.accept.astype(bool)
    index_names = [ind for ind in df.index.names if ind != 'index']
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
    # number of threads in each lstg
    thread_counter = max_offers.reset_index('thread').thread.groupby('lstg').max()
    # unstack by thread and fill with last value
    max_offers = max_offers.unstack(level='thread').groupby('lstg').ffill()

    if role == 'byr':
        offer_counter = df.byr & ~df.reject
    else:
        offer_counter = ~df.byr & ~df.reject
    offer_counter = offer_counter.unstack(level='thread')
    thread_counter = thread_counter.reindex(index=offer_counter.index, level='lstg')
    print(thread_counter)
    # initialize dictionaries for later concatenation
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
    # concat into series and return
    return collapse_dict(count, index_names), collapse_dict(best, index_names)


def get_lstg_time_feats(events):
    # create dataframe for variable creation
    ordered = events.sort_values(['lstg', 'clock', 'index', 'censored']).drop(
        ['message', 'price'], axis=1)
    # identify listings with multiple threads
    threads = ordered.reset_index().groupby('lstg')['thread'].nunique()
    check = threads > 1
    subset = ordered.loc[check[check].index].set_index(['clock'], append=True)
    subset = subset.reorder_levels(['lstg', 'thread', 'clock', 'index'])
    # add features for open offers
    tf = pd.DataFrame()
    for role in ['slr', 'byr']:
        cols = [role + c for c in ['_offers', '_best']]
        for is_open in [False, True]:
            if is_open:
                cols = [c + '_open' for c in cols]
            tf[cols[0]], tf[cols[1]] = add_lstg_time_feats(
                subset, role, is_open)
    tf['thread_count'] = thread_count(subset)
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
    tf_lstg = get_lstg_time_feats(events)

    # save
    dump(tf_lstg, FEATS_DIR + '%d_tf_lstg.gz' % num)

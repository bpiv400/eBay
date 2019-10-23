from constants import *
from processing.b_feats.time_funcs import *


def collapse_dict(feat_dict, index_names, meta=False):
    if not meta:
        remaining = [ind for ind in index_names if ind != 'thread']
        df = pd.concat(feat_dict, names=['thread'] + remaining)
    else:
        remaining = [ind for ind in index_names if ind != 'lstg_counter']
        df = pd.concat(feat_dict, names=['lstg_counter'] + remaining)
    df = df.reorder_levels(index_names).sort_index()
    return df


def get_quantiles(df, l, featname):
    # initialize output dataframe
    df = df.copy()
    df = df.drop(columns='thread')
    converter = df[['lstg_counter']]
    converter = converter.set_index('lstg_counter', append=True).reset_index('lstg', drop=False)['lstg']
    drop_levels = [ind_level for ind_level in converter.index.names if ind_level != 'lstg_counter'
                   and ind_level not in l]
    converter.index = converter.index.droplevel(drop_levels)
    converter = converter.drop_duplicates(keep='first')
    # subset to 1 entry per lstg per hierarchy group

    accepts = df.reset_index(drop=False)
    accepts = accepts[[featname, 'lstg_counter'] + l]
    accepts = accepts.groupby(by=l + ['lstg_counter']).max()[featname]

    # total lstgs
    total_lstgs = df.reset_index().groupby(by=l).max()['lstg_counter']
    if len(l) == 1:
        total_lstgs = total_lstgs.reindex(accepts.index, level=l[0])
    else:
        total_lstgs = total_lstgs.reindex(accepts.index)

    quants = dict()
    # loop over quantiles
    for n in range(int(total_lstgs.max()) + 1):
        cut = accepts.loc[total_lstgs >= n].drop(n, level='lstg_counter')
        rel_groups = cut.index.droplevel('lstg_counter').drop_duplicates()
        cut = cut.groupby(by=l)
        partial = pd.DataFrame(index=rel_groups)
        for q in QUANTILES:
            tfname = '_'.join([l[-1], featname, str(int(100 * q))])
            partial[tfname] = cut.quantile(q=q, interpolation='lower').fillna(0)
        assert not partial.isna().any().any()
        quants[n] = partial
    # combine
    output = collapse_dict(quants, l + ['lstg_counter'], meta=True)
    assert output.index.is_unique
    output = output.join(converter)
    output = output.reset_index('lstg_counter', drop=True).set_index('lstg', append=False,
                                                                     drop=True)
    return output


def get_cat_feats(events, levels=None, feat_ind=None):
    # helper features
    df = events.copy()
    df['clock'] = pd.to_datetime(df.clock, unit='s', origin=START)
    df['lstg_ind'] = (df.index.get_level_values('index') == 0).astype(bool)
    df['base'] = (df.index.get_level_values('thread') == 0).astype(bool)
    df['thread'] = (df.index.get_level_values('index') == 1).astype(bool)
    df['thread'] = (df.thread & ~df.base).astype(bool)
    df['slr_offer'] = (~df.byr & ~df.reject & ~df.lstg_ind & ~df.accept & ~df.base).astype(bool)
    df['byr_offer'] = df.byr & ~df.reject & ~df.accept
    df['lstg_id'] = df.index.get_level_values('lstg').astype(np.int64)

    if feat_ind == 1:
        return get_cat_lstg_counts(df, levels)
    elif feat_ind == 2:
        return get_cat_accepts(df, levels)
    elif feat_ind == 3:
        return get_cat_con(df, levels)
    elif feat_ind == 4:
        return get_cat_delay(df, levels)
    elif feat_ind == 5:
        return get_cat_start_price(df, levels)
    elif feat_ind == 6:
        return get_cat_byr_hist(df, levels)
    elif feat_ind == 7:
        return get_cat_arrival(df, levels)
    else:
        raise NotImplementedError('feature index must be between 1 and 7')


def get_cat_accepts(events, levels):
    # loop over hierarchy, exlcuding lstg
    events['accept_norm'] = events.price[events.accept & ~events.flag] / events.start_price
    tf = events[['clock']].xs(0, level='thread', drop_level=True)
    tf = tf.drop(columns=['clock'])
    tf = tf.xs(0, level='index', drop_level=True)
    tf = tf.reset_index('lstg', drop=False)
    tf = tf.reset_index(drop=True)
    tf = tf.set_index('lstg', drop=True)

    for i in range(len(levels)):
        curr_levels = levels[: i + 1]
        events['lstg_counter'] = events['lstg_id'].groupby(by=curr_levels).transform(
            lambda x: x.factorize()[0].astype(np.int64)
        )
        # quantiles of (normalized) accept price over 30-day window
        quants = get_quantiles(events, curr_levels, 'accept_norm')
        tf = tf.join(quants)
        # for identical timestamps
        cols = [c for c in tf.columns if c.startswith(levels[-1])]
    # collapse to lstg
    return tf.sort_index()


def get_cat_con(events, levels):
    pass


def get_cat_delay(events, levels):
    pass


def get_cat_start_price(events, levels):
    pass


def get_cat_byr_hist(events, levels):
    pass


def get_cat_arrival(events, levels):
    pass


def get_cat_lstg_counts(events, levels):
    # initialize output dataframe
    tf = events[['clock']].xs(0, level='thread', drop_level=True)
    print(events.columns)
    tf = tf.drop(columns=['clock'])
    tf = tf.xs(0, level='index', drop_level=True)
    tf = tf.reset_index('lstg', drop=False)
    tf = tf.reset_index(drop=True)
    tf = tf.set_index('lstg', drop=True)

    # loop over hierarchy, exlcuding lstg
    for i in range(len(levels)):
        curr_levels = levels[: i+1]
        # open listings
        tfname = '_'.join([curr_levels[-1], 'lstgs_open'])
        events = events.sort_values(['clock', 'censored'] + curr_levels)
        tf[tfname] = open_lstgs(events, curr_levels)
        # count features grouped by current level
        ct_feats = events[['lstg_ind', 'thread', 'slr_offer',
                       'byr_offer', 'accept']].groupby(by=curr_levels).sum()
        ctl_feats = events[['lstg_ind', 'thread', 'slr_offer',
                        'byr_offer', 'accept']].groupby(by=curr_levels + ['lstg']).sum()
        ct_feats = ct_feats - ctl_feats
        per_lstg_feats = ['accept', 'slr_offer', 'byr_offer', 'thread']
        for feat in per_lstg_feats:
            ct_feats[feat] = ct_feats[feat] / (ct_feats['lstg_ind'] + 1)
        ct_feats = ct_feats.rename(lambda x: '_'.join([curr_levels[-1], x]) + 's', axis=1)
        tf = tf.join(ct_feats)
        tf.index = tf.index.droplevel(level=curr_levels)
        cols = [c for c in tf.columns if c.startswith(levels[-1])]

    # collapse to lstg
    tf = tf.rename(lambda curr_name:
                   curr_name if 'lstg_ind' not in curr_name else curr_name.replace('_ind', ''),
                   axis=1)
    return tf.sort_index()


def get_cat_time_feats(events, levels):
    # initialize output dataframe
    tf = events[['clock']]
    # dataframe for variable calculations
    df = events.copy()
    df['clock'] = pd.to_datetime(df.clock, unit='s', origin=START)
    df['lstg_ind'] = (df.index.get_level_values('index') == 0).astype(bool)
    df['base'] = (df.index.get_level_values('thread') == 0).astype(bool)
    df['thread'] = (df.index.get_level_values('index') == 1).astype(bool)
    df['thread'] = (df.thread & ~df.base).astype(bool)
    df['slr_offer'] = (~df.byr & ~df.reject & ~df.lstg_ind & ~df.accept & ~df.base).astype(bool)
    df['byr_offer'] = df.byr & ~df.reject & ~df.accept
    df['accept_norm'] = df.price[df.accept & ~df.flag] / df.start_price
    df['lstg_id'] = df.index.get_level_values('lstg').astype(np.int64)

    # loop over hierarchy, exlcuding lstg
    for i in range(len(levels)):
        l = levels[: i+1]
        if len(l) != len(levels):
            others = levels[i+1:]
        else:
            others = []
        print(l[-1])
        print(l)
        df['lstg_counter'] = df['lstg_id'].groupby(by=l).transform(
            lambda x: x.factorize()[0].astype(np.int64)
        )
        # sort by levels
        curr_order = l + ['lstg', 'thread', 'index'] + others
        df = df.sort_values(['clock', 'censored'] + l).reorder_levels(curr_order)
        tf = tf.reorder_levels(curr_order).reindex(df.index)
        # sanity check
        pre_index = tf.index
        # open listings
        tfname = '_'.join([l[-1], 'lstgs_open'])
        tf[tfname] = open_lstgs(df, l)
        # count features grouped by current level
        ct_feats = df[['lstg_ind', 'thread', 'slr_offer',
                       'byr_offer', 'accept']].groupby(by=l).sum()
        ctl_feats = df[['lstg_ind', 'thread', 'slr_offer',
                       'byr_offer', 'accept']].groupby(by=l + ['lstg']).sum()
        if l[-1] == 'meta':
            print(ctl_feats.xs(1, level='meta')['thread'])
        ct_feats = ct_feats - ctl_feats
        ct_feats = ct_feats.rename(lambda x:'_'.join([l[-1], x]) + 's', axis=1)
        ct_feats = ct_feats.astype(np.int64).reorder_levels(l + ['lstg'])
        ct_feats = ct_feats.reindex(tf.index)
        tf = tf.join(ct_feats)
        if l[-1] == 'meta':
            print(tf['meta_threads'].xs(0, level='thread').xs(1, level='meta').xs(0, level='index'))
        # quantiles of (normalized) accept price over 30-day window
        quants = get_quantiles(df, l, 'accept_norm')
        quants = quants.reorder_levels(l + ['lstg'])
        quants = quants.reindex(tf.index)
        tf = tf.join(quants)
        # for identical timestamps
        cols = [c for c in tf.columns if c.startswith(levels[-1])]
        tf = tf.reindex(pre_index)
        tf[cols] = tf[['clock'] + cols].groupby(
            by=l + ['clock', 'lstg']).transform('last')
    # collapse to lstg
    tf = tf.xs(0, level='index').reset_index(levels + ['thread'], drop=True).drop('clock', axis=1)
    tf = tf.rename(lambda curr_name:
                   curr_name if 'lstg_ind' not in curr_name else curr_name.replace('_ind', ''),
                   axis=1)
    return tf.sort_index()


def create_obs(df, isStart, cols):
    toAppend = pd.DataFrame(index=df.index, columns=['index'] + cols)
    for c in ['accept', 'message']:
        if c in cols:
            toAppend[c] = False
    if isStart:
        toAppend.loc[:, 'reject'] = False
        toAppend.loc[:, 'index'] = 0
        toAppend.loc[:, 'censored'] = False
        toAppend.loc[:, 'price'] = df.start_price
        toAppend.loc[:, 'clock'] = df.start_time
    else:
        toAppend.loc[:, 'reject'] = True
        toAppend.loc[:, 'index'] = 1
        toAppend.loc[:, 'censored'] = True
        toAppend.loc[:, 'price'] = np.nan
        toAppend.loc[:, 'clock'] = df.end_time
    return toAppend.set_index('index', append=True)


def expand_index(df, levels):
    df.set_index(levels, append=True, inplace=True)
    idxcols = levels + ['lstg', 'thread']
    if 'index' in df.index.names:
        idxcols += ['index']
    df = df.reorder_levels(idxcols)
    df.sort_values(idxcols, inplace=True)
    return df


def add_start_end(offers, L, levels):
    # listings dataframe
    lstgs = L[levels + ['start_date', 'end_time', 'start_price']].copy()
    lstgs['thread'] = 0
    lstgs.set_index('thread', append=True, inplace=True)
    lstgs = expand_index(lstgs, levels)
    lstgs['start_time'] = lstgs.start_date * 60 * 60 * 24
    lstgs.drop('start_date', axis=1, inplace=True)
    lstgs = lstgs.join(offers['accept'].groupby('lstg').max())
    # create data frames to append to offers
    cols = list(offers.columns)
    start = create_obs(lstgs, True, cols)
    end = create_obs(lstgs[lstgs.accept != 1], False, cols)
    # append to offers
    offers = offers.append(start, sort=True).append(end, sort=True)
    # sort
    return offers.sort_index()


def init_offers(L, T, O, levels):
    offers = O.join(T['start_time'])
    for c in ['accept', 'reject', 'censored', 'message']:
        if c in offers:
            offers[c] = offers[c].astype(np.bool)
    offers['clock'] += offers['start_time']
    offers.drop('start_time', axis=1, inplace=True)
    offers = offers.join(L[levels])
    offers = expand_index(offers, levels)
    return offers


def create_events(L, T, O, levels):
    # initial offers data frame
    offers = init_offers(L, T, O, levels)
    # add start times and expirations for unsold listings
    events = add_start_end(offers, L, levels)
    # add features for later use
    events['byr'] = events.index.isin(IDX['byr'], level='index')
    events = events.join(L[['flag', 'start_price']])
    return events



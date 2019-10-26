from constants import *
from processing.b_feats.time_funcs import *


def prep_tf(events):
    tf = events[['clock']].xs(0, level='thread', drop_level=True)
    tf = tf.drop(columns=['clock'])
    tf = tf.xs(0, level='index', drop_level=True)
    tf = tf.reset_index('lstg', drop=False)
    tf = tf.reset_index(drop=True)
    tf = tf.set_index('lstg', drop=True)
    return tf


def collapse_dict(feat_dict, index_names, meta=False):
    if not meta:
        remaining = [ind for ind in index_names if ind != 'thread']
        df = pd.concat(feat_dict, names=['thread'] + remaining)
    else:
        remaining = [ind for ind in index_names if ind != 'lstg_counter']
        df = pd.concat(feat_dict, names=['lstg_counter'] + remaining)
    df = df.reorder_levels(index_names).sort_index()
    return df


def quant_vector_index(quant_vector, l, name):
    quant_vector = quant_vector.reset_index(drop=False)
    quant_vector = quant_vector[[name, 'lstg_counter'] + l]
    quant_vector = quant_vector.set_index(l + ['lstg_counter'], append=False,
                                          drop=True)
    quant_vector = quant_vector[name]
    return quant_vector


def prep_quantiles(df, l, featname):
    print('')
    print(featname)
    print(featname == 'arrival_rate')
    if featname == 'accept_norm':
        quant_vector = df.reset_index(drop=False)
        quant_vector = quant_vector[[featname, 'lstg_counter'] + l]
        quant_vector = quant_vector.groupby(by=l + ['lstg_counter']).max()[featname]
    elif featname == 'first_offer' or featname == 'byr_hist':
        quant_vector = df.xs(1, level='index')
        quant_vector = quant_vector_index(quant_vector, l, featname)
    elif featname == 'slr_delay' or featname == 'byr_delay':
        if featname == 'slr_delay':
            other_turn = df.index.get_level_values('index').isin([3, 5])
        else:
            other_turn = df.index.get_level_values('index').isin([2, 4, 6])
        # removing the other turn from the distribution
        df.loc[other_turn, 'delay'] = np.NaN
        # removing auto rejects from the distribution
        auto_rej = df['delay'] == 1
        df.loc[auto_rej, 'delay'] = np.NaN
        quant_vector = quant_vector_index(df, l, 'delay')
        quant_vector = quant_vector.rename(featname)
    elif featname == 'start_price_pctile' or featname == 'arrival_rate':
        quant_vector = df.xs(0, level='thread').xs(0, level='index')
        quant_vector = quant_vector_index(quant_vector, l, featname)
        print('quant vec')
        print(quant_vector)
    elif featname == 'byr_hist':
        quant_vector = df.xs(1, level='index')
        quant_vector = quant_vector_index(quant_vector, l, featname)
    else:
        raise NotImplementedError()
    return quant_vector


def get_perc(df, l, num, denom, name=None):
    df = df.copy()
    level_group = df[[num, denom]].groupby(by=l).sum()
    lstg_group = df[[num, denom]].groupby(by=l + ['lstg']).sum()
    num_level = level_group[num]
    num_lstg = lstg_group[num]
    num_ser = num_level - num_lstg

    if name == 'bin':
        assert num_ser.max() <= 1  # sanity check

    denom_level = level_group[denom]
    denom_lstg = lstg_group[denom]
    denom_ser = denom_level - denom_lstg

    output_ser = (num_ser / denom_ser).fillna(0)
    assert not np.any(np.isinf(output_ser.values))
    output_ser.index = output_ser.index.droplevel(level=l)
    output_ser = output_ser.rename('{}_{}'.format(l[-1], name))
    return output_ser


def get_expire_perc(df, l, byr=False):
    df = df.copy()
    if byr:
        other_turn = df.index.get_level_values('index').isin([2, 4, 6])
        name = 'byr_expire'
    else:
        other_turn = df.index.get_level_values('index').isin([3, 5])
        name = 'slr_expire'
    df.loc[other_turn, 'delay'] = np.NaN
    df['expire'] = (df['delay'] == 1).astype(bool)
    df['curr_offer'] = df['delay'].notna()
    return get_perc(df, l, 'expire', 'curr_offer', name=name)


def get_quantiles(df, l, featname):
    # initialize output dataframe
    df = df.copy()
    df['lstg_counter'] = df['lstg_id'].groupby(by=l).transform(
        lambda x: x.factorize()[0].astype(np.int64))
    df = df.drop(columns='thread')
    converter = df[['lstg_counter']]
    converter = converter.set_index('lstg_counter', append=True)
    converter = converter.reset_index('lstg', drop=False)['lstg']
    drop_levels = [ind_level for ind_level in converter.index.names if ind_level != 'lstg_counter'
                   and ind_level not in l]
    converter.index = converter.index.droplevel(drop_levels)
    converter = converter.drop_duplicates(keep='first')

    # subset to 1 entry per lstg per hierarchy group
    quant_vector = prep_quantiles(df, l, featname)
    # total lstgs
    total_lstgs = df.reset_index().groupby(by=l).max()['lstg_counter']
    if len(l) == 1:
        total_lstgs = total_lstgs.reindex(quant_vector.index, level=l[0])
    else:
        total_lstgs = total_lstgs.reindex(quant_vector.index)

    quants = dict()
    # loop over quantiles
    for n in range(int(total_lstgs.max()) + 1):
        cut = quant_vector.loc[total_lstgs >= n].drop(n, level='lstg_counter')
        rel_groups = cut.index.droplevel('lstg_counter').drop_duplicates()
        cut = cut.groupby(by=l)
        partial = pd.DataFrame(index=rel_groups)
        for q in QUANTILES:
            tfname = '_'.join([l[-1], featname, str(int(100 * q))])
            partial[tfname] = cut.quantile(q=q, interpolation='lower')
            partial[tfname] = partial[tfname].fillna(0)
        quants[n] = partial
    # combine
    output = collapse_dict(quants, l + ['lstg_counter'], meta=True)
    assert output.index.is_unique
    output = output.join(converter, how='right')
    output = output.reset_index('lstg_counter', drop=True).set_index('lstg', append=False,
                                                                     drop=True)
    output = output.fillna(0)
    assert not output.isna().any().any()
    return output


def get_cat_feats(events, levels=None, feat_ind=None):
    # helper features
    df = events.copy()
    df = df.sort_index()
    df['clock'] = pd.to_datetime(df.clock, unit='s', origin=START)
    df['lstg_ind'] = (df.index.get_level_values('index') == 0).astype(bool)
    df['base'] = (df.index.get_level_values('thread') == 0).astype(bool)
    df['thread'] = (df.index.get_level_values('index') == 1).astype(bool)
    df['thread'] = (df.thread & ~df.base).astype(bool)
    df['slr_offer'] = (~df.byr & ~df.reject & ~df.lstg_ind & ~df.accept & ~df.base).astype(bool)
    df['byr_offer'] = df.byr & ~df.reject & ~df.accept
    df['lstg_id'] = df.index.get_level_values('lstg').astype(np.int64)

    df['offer_norm'] = df.price / df.start_price
    real_threads = (df.index.get_level_values('thread') != 0).astype(bool)
    first_offer = (df.index.get_level_values('index') == 1).astype(bool)
    df['bin'] = (first_offer & real_threads & (df.offer_norm == 1).astype(bool)).astype(bool)
    first_offer = first_offer & real_threads & ~df.flag & ~df.bin
    df['first_offer'] = (df.offer_norm[first_offer]).astype(np.float64)
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


def get_cat_lstg_counts(events, levels):
    # initialize output dataframe
    tf = prep_tf(events)

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


def get_cat_accepts(events, levels):
    # loop over hierarchy, excluding lstg
    events['accept_norm'] = events.price[events.accept & ~events.flag & ~events.bin] / events.start_price
    tf = prep_tf(events)

    for i in range(len(levels)):
        curr_levels = levels[: i + 1]
        # quantiles of (normalized) accept price over 30-day window
        quants = get_quantiles(events, curr_levels, 'accept_norm')
        tf = tf.join(quants)
        # for identical timestamps
        cols = [c for c in tf.columns if c.startswith(levels[-1])]
    # collapse to lstg
    return tf.sort_index()


def get_cat_con(events, levels):
    # loop over hierarchy, exlcuding lstg
    # bin / thread (excluding curr)
    tf = prep_tf(events)

    for i in range(len(levels)):
        curr_levels = levels[: i + 1]
        events['lstg_counter'] = events['lstg_id'].groupby(by=curr_levels).transform(
            lambda x: x.factorize()[0].astype(np.int64)
        )
        bin = get_perc(events,  curr_levels, 'bin', 'lstg_ind', name='bin')
        tf = tf.join(bin)
        # quantiles of (normalized) accept price over 30-day window
        quants = get_quantiles(events, curr_levels, 'first_offer')
        tf = tf.join(quants)
        # for identical timestamps
        cols = [c for c in tf.columns if c.startswith(levels[-1])]
    # collapse to lstg
    return tf.sort_index()


def get_cat_delay(events, levels):
    tf = prep_tf(events)
    clock_df = events.clock.unstack()
    delay = pd.DataFrame(index=clock_df.index)
    delay[0] = np.NaN # excluding turn 0
    for i in range(1, 8):
        if i in clock_df.columns:
            delay[i] = (clock_df[i] - clock_df[i - 1]).dt.total_seconds()
            if i in [2, 4, 6]:
                max_delay = MAX_DELAY['slr']
                censored = delay[i] > MAX_DELAY['slr']
            elif i in [3, 5]:
                censored = delay[i] > MAX_DELAY['byr']
                max_delay = MAX_DELAY['byr']
            else: # excluding turn 1 and turn 7
                censored = delay.index
                max_delay = 1
            delay.loc[censored, i] = np.NaN
            delay[i] = delay[i] / max_delay
    events['delay'] = delay.rename_axis('index', axis=1).stack()
    # excluding auto rejections
    events.loc[events['delay'] == 0, 'delay'] = np.NaN
    # excluding censored rejections
    events.loc[events['censored'], 'delay'] = np.NaN
    for i in range(len(levels)):
        curr_levels = levels[: i + 1]
        events['lstg_counter'] = events['lstg_id'].groupby(by=curr_levels).transform(
            lambda x: x.factorize()[0].astype(np.int64)
        )
        tf = tf.join(get_quantiles(events, curr_levels, 'slr_delay'))
        tf = tf.join(get_quantiles(events, curr_levels, 'byr_delay'))
        tf = tf.join(get_expire_perc(events, curr_levels, byr=False))
        tf = tf.join(get_expire_perc(events, curr_levels, byr=True))
    # collapse to lstg
    return tf.sort_index()


def get_cat_start_price(events, levels):
    return get_cat_quantiles_wrapper(events, levels, 'start_price_pctile')


def get_cat_byr_hist(events, levels):
    events.loc[events.base, 'byr_hist'] = np.NaN
    events.byr_hist = events.byr_hist.astype(np.float)
    return get_cat_quantiles_wrapper(events, levels, 'byr_hist')


def get_cat_arrival(events, levels):
    return get_cat_quantiles_wrapper(events, levels, 'arrival_rate')


def get_cat_quantiles_wrapper(events, levels, featname):
    tf = prep_tf(events)
    for i in range(len(levels)):
        curr_levels = levels[: i + 1]
        tf = tf.join(get_quantiles(events, curr_levels, featname))
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
    offers = O.join(T[['start_time', 'byr_hist']])
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
    events = events.join(L[['flag', 'start_price', 'arrival_rate']])
    return events



import numpy as np
import pandas as pd
from utils import feat_to_pctile
from constants import IDX, MAX_DELAY_TURN, DAY
from processing.const import START
from featnames import BYR_HIST, DELAY, LSTG, THREAD, INDEX, CLOCK, ACCEPT, \
    BYR, START_TIME, START_PRICE, START_DATE, END_TIME, REJECT

QUANTILES = [25, 50, 75, 100]  # quantiles of feature distribution


def prep_tf(events):
    tf = events[[CLOCK]].xs(0, level=THREAD, drop_level=True)
    tf = tf.drop(columns=[CLOCK])
    tf = tf.xs(0, level=INDEX, drop_level=True)
    tf = tf.reset_index(LSTG, drop=False)
    tf = tf.reset_index(drop=True)
    tf = tf.set_index(LSTG, drop=True)
    return tf


def collapse_dict(feat_dict, index_names, meta=False):
    if not meta:
        remaining = [ind for ind in index_names if ind != THREAD]
        df = pd.concat(feat_dict, names=[THREAD] + remaining)
    else:
        remaining = [ind for ind in index_names if ind != 'lstg_counter']
        df = pd.concat(feat_dict, names=['lstg_counter'] + remaining)
    df = df.reorder_levels(index_names).sort_index()
    return df


def quant_vector_index(quant_vector, elems, name, new=False):
    lstg_ind = 'lstg_id' if new else 'lstg_counter'
    quant_vector = quant_vector.reset_index(drop=False)
    quant_vector = quant_vector[[name, lstg_ind] + elems]
    quant_vector = quant_vector.set_index(elems + [lstg_ind], append=False,
                                          drop=True)
    quant_vector = quant_vector[name]
    return quant_vector


def prep_quantiles(df, elems, featname, new=False):
    lstg_ind = 'lstg_id' if new else 'lstg_counter'
    if featname == 'accept_norm':
        quant_vector = df.reset_index(drop=False)
        quant_vector = quant_vector[[featname, lstg_ind] + elems]
        quant_vector = quant_vector.groupby(by=elems + [lstg_ind]).max()[featname]
    elif featname == 'first_offer' or featname == BYR_HIST:
        quant_vector = df.xs(1, level=INDEX)
        quant_vector = quant_vector_index(quant_vector, elems, featname, new=new)
    elif featname == 'slr_delay' or featname == 'byr_delay':
        if featname == 'slr_delay':
            other_turn = df.index.get_level_values(INDEX).isin([3, 5])
        else:
            other_turn = df.index.get_level_values(INDEX).isin([2, 4, 6])
        # removing the other turn from the distribution
        df.loc[other_turn, DELAY] = np.NaN
        # removing expiration rejects from the distribution
        auto_rej = df[DELAY] == 1
        df.loc[auto_rej, DELAY] = np.NaN
        quant_vector = quant_vector_index(df, elems, DELAY, new=new)
        quant_vector = quant_vector.rename(featname)
    elif featname == 'start_price_pctile' or featname == 'arrival_rate':
        quant_vector = df.xs(0, level=THREAD).xs(0, level=INDEX)
        quant_vector = quant_vector_index(quant_vector, elems, featname, new=new)
    elif featname == BYR_HIST:
        quant_vector = df.xs(1, level=INDEX)
        quant_vector = quant_vector_index(quant_vector, elems, featname, new=new)
    else:
        raise NotImplementedError()
    return quant_vector


def get_perc(df, elems, num, denom, name=None):
    df = df.copy()
    if 'msg_rate' not in name:
        df[num] = df[num].astype(int)
        df[denom] = df[denom].astype(int)
    level_group = df[[num, denom]].groupby(by=elems).sum()
    lstg_group = df[[num, denom]].groupby(by=elems + [LSTG]).sum()
    num_level = level_group[num]
    num_lstg = lstg_group[num]
    num_ser = num_level - num_lstg

    if name == 'bin':
        assert num_lstg.max() <= 1  # sanity check

    denom_level = level_group[denom]
    denom_lstg = lstg_group[denom]
    denom_ser = denom_level - denom_lstg

    output_ser = (num_ser / denom_ser).fillna(0)
    assert not np.any(np.isinf(output_ser.values))
    output_ser.index = output_ser.index.droplevel(level=elems)
    output_ser = output_ser.rename('{}_{}'.format(elems[-1], name))
    return output_ser


def get_expire_perc(df, elems, byr=False):
    df = df.copy()
    if byr:
        other_turn = df.index.get_level_values(INDEX).isin([2, 4, 6])
        name = 'byr_expire'
    else:
        other_turn = df.index.get_level_values(INDEX).isin([3, 5])
        name = 'slr_expire'
    df.loc[other_turn, DELAY] = np.NaN
    df['expire'] = (df[DELAY] == 1).astype(bool)
    df['curr_offer'] = df[DELAY].notna()
    return get_perc(df, elems, 'expire', 'curr_offer', name=name)


def original_quantiles(quant_vector=None, total_lstgs=None, featname=None,
                       elems=None, converter=None):
    quants = dict()
    # loop over quantiles
    for n in range(int(total_lstgs.max()) + 1):
        cut = quant_vector.loc[total_lstgs >= n].drop(n, level='lstg_counter')
        rel_groups = cut.index.droplevel('lstg_counter').drop_duplicates()
        cut = cut.groupby(by=elems)
        partial = pd.DataFrame(index=rel_groups)
        for q in QUANTILES:
            tfname = '_'.join([elems[-1], featname, str(q)])
            partial[tfname] = cut.quantile(q=q, interpolation='lower')
            partial[tfname] = partial[tfname].fillna(0)
        quants[n] = partial
    # combine
    output = collapse_dict(quants, elems + ['lstg_counter'], meta=True)
    assert output.index.is_unique
    output = output.join(converter, how='right')
    output = output.reset_index('lstg_counter', drop=True)
    output = output.set_index(LSTG, append=False, drop=True)
    output = output.fillna(0)
    assert not output.isna().any().any()
    return output


def inplace_quantiles(df):
    """
    Assumptions
    1. df contains targ (vector of interest)
    2. df contains lstg (vector giving lstg id)
    3. Some or all of the entries in targ may be nans
    4. df is sorted in ascending order of targ (with nans at back)

    :param df: pandas.DataFrame
    :return:
    """
    total_count = len(df)
    nan_count = df['targ'].isna().sum()
    filled_count = total_count - nan_count
    # edge case of all nans
    if filled_count == 0:
        lstgs = df[LSTG].unique()
        lstgs = pd.Index(lstgs, name=LSTG)
        cols = [str(i) for i in QUANTILES]
        res_dict = pd.DataFrame(0, columns=cols, index=lstgs)
        return res_dict
    last_filled = filled_count - 1
    targs = df['targ'].values
    lstgs = df[LSTG].values
    # compute indices of quantiles in full distribution
    full_quant_inds = {q: int(q * last_filled / 100) for q in QUANTILES}
    lstg_ind_dict = dict()
    lstg_count_dict = dict()
    # populate dictionary of lstgs with lists containing indices corresponding
    # to each lstg and a count of the number of non-nan elements
    for i, (curr_targ, curr_lstg) in enumerate(zip(targs, lstgs)):
        if curr_lstg not in lstg_ind_dict:
            lstg_ind_dict[curr_lstg] = list()
            lstg_count_dict[curr_lstg] = 0
        lstg_ind_dict[curr_lstg].append(i)
        if i <= last_filled:
            lstg_count_dict[curr_lstg] += 1

    res_dict = {i: list() for i in QUANTILES}
    lstg_order = []
    # iterate over all lstgs
    for curr_lstg, curr_inds in lstg_ind_dict.items():
        # establish order in which we iterated over lstgs for index population later
        lstg_order.append(curr_lstg)
        non_nan_count = lstg_count_dict[curr_lstg]
        # iterate over quantiles
        for q in res_dict.keys():
            q_normal = q / 100
            # handle edge case where no elements are removed for the current lstg
            if non_nan_count == 0:
                res_dict[q].append(targs[full_quant_inds[q]])
            # handle edge case where all elements are removed for the current lstg
            elif non_nan_count == filled_count:
                res_dict[q].append(0)
            else:
                # set a guess for the index of the quantile assuming no values
                # occur before the quantile
                q_g = int((filled_count - non_nan_count - 1) * q_normal)
                # for each index that occurs before the quantile, increment the quantile
                for ind in curr_inds:
                    if ind <= q_g:
                        q_g += 1
                    else:
                        break
                q_val = targs[q_g]
                # if np.isnan(q_val):
                #     raise RuntimeError('hmm... we gotta problem')
                res_dict[q].append(q_val)
    # convert output into a dataframe
    res_dict = pd.DataFrame.from_dict(res_dict, orient='columns')
    res_dict = res_dict.rename(columns=lambda col: str(col))
    lstg_order = pd.Index(lstg_order, name=LSTG)
    res_dict.index = lstg_order
    return res_dict


def fast_quantiles(df, elems, featname):
    # initialize output dataframe
    df = df.copy()
    df = df.drop(columns=THREAD)
    # subset to minimal entries per lstg
    quant_vector = prep_quantiles(df, elems, featname, new=True)
    # sort
    quant_vector = quant_vector.sort_values(na_position='last')
    # name feature series
    quant_vector.name = 'targ'
    # reset index and add unique counter
    quant_vector = quant_vector.reset_index(drop=False)
    quant_vector.index.name = 'counter'
    quant_vector = quant_vector.set_index(elems, append=True, drop=True)
    # add lstg column
    quant_vector = quant_vector.rename(columns={'lstg_id': LSTG})
    # group
    vector_groups = quant_vector.groupby(by=elems)
    acc = list()
    # iterate over groups
    for index_subset in vector_groups.groups.values():
        subset = quant_vector.loc[index_subset].copy()
        subset_quants = inplace_quantiles(subset)
        acc.append(subset_quants)
    # concatenate results for each group
    output = pd.concat(acc, axis=0, sort=False)
    # append feature and level name to the column
    level_name = elems[-1]
    output = output.rename(columns=lambda q: '{}_{}_{}'.format(level_name,
                                                               featname, q))
    output = output.sort_index()
    return output


def get_quantiles(df, elems, featname):
    # initialize output dataframe
    df = df.copy()
    df['lstg_counter'] = df['lstg_id'].groupby(by=elems).transform(
        lambda x: x.factorize()[0].astype(np.int64))
    df = df.drop(columns=THREAD)
    converter = df[['lstg_counter']]
    converter = converter.set_index('lstg_counter', append=True)
    converter = converter.reset_index(LSTG, drop=False)[LSTG]
    drop_levels = [ind_level for ind_level in converter.index.names if ind_level != 'lstg_counter'
                   and ind_level not in elems]
    converter.index = converter.index.droplevel(drop_levels)
    converter = converter.drop_duplicates(keep='first')

    # subset to 1 entry per lstg per hierarchy group
    quant_vector = prep_quantiles(df, elems, featname)
    # total lstgs
    total_lstgs = df.reset_index().groupby(by=elems).max()['lstg_counter']
    if len(elems) == 1:
        total_lstgs = total_lstgs.reindex(quant_vector.index, level=elems[0])
    else:
        total_lstgs = total_lstgs.reindex(quant_vector.index)

    return original_quantiles(quant_vector=quant_vector,
                              total_lstgs=total_lstgs,
                              featname=featname,
                              elems=elems,
                              converter=converter)


def make_helper_feats(events):
    df = events.sort_index()
    df[CLOCK] = pd.to_datetime(df[CLOCK], unit='s', origin=START)
    df['lstg_ind'] = (df.index.get_level_values(INDEX) == 0).astype(bool)
    df['base'] = (df.index.get_level_values('thread') == 0).astype(bool)
    df[THREAD] = (df.index.get_level_values(INDEX) == 1).astype(bool)
    df[THREAD] = (df[THREAD] & ~df.base).astype(bool)
    df['slr_offer'] = (~df[BYR] & ~df[REJECT] & ~df.lstg_ind & ~df[ACCEPT] & ~df.base).astype(bool)
    df['byr_offer'] = df[BYR] & ~df[REJECT] & ~df[ACCEPT]
    df['lstg_id'] = df.index.get_level_values(LSTG).astype(np.int64)
    df['offer_norm'] = df.price / df[START_PRICE]
    real_threads = (df.index.get_level_values(THREAD) != 0).astype(bool)
    first_offer = (df.index.get_level_values(INDEX) == 1).astype(bool)
    df['bin'] = (first_offer & real_threads & (df.offer_norm == 1).astype(bool)).astype(bool)
    df['bin'] = (df['bin'] & df[ACCEPT]).astype(bool)
    first_offer = first_offer & real_threads & ~df.bin
    df['first_offer'] = (df.offer_norm[first_offer]).astype(np.float64)
    return df


def get_all_cat_feats(events, levels):
    # helper features
    df = make_helper_feats(events)
    tf = None
    for i in range(1, 9):
        temp = get_cat_feat(df, levels=levels, feat_ind=i)
        if i == 1:
            tf = temp
        else:
            tf = tf.join(temp)
    return tf.sort_index()


def get_cat_feat(events, levels=None, feat_ind=None):
    df = events.copy()
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
    elif feat_ind == 8:
        return get_cat_msg_rate(df, levels)
    else:
        raise NotImplementedError('feature index must be between 1 and 7')


def get_msg_rate(events, levels, turn):
    events = events[['message']].copy()
    other_turns = ~events.index.get_level_values(INDEX).isin([turn])
    events.loc[other_turns, 'message'] = np.NaN

    events['is_turn'] = (~events.message.isna())
    events.loc[:, 'is_turn'] = events['is_turn'].astype(bool)
    events.loc[:, 'is_turn'] = events['is_turn'].astype(int)
    name = 'msg_rate_{}'.format(turn)
    msg_rate = get_perc(events, levels, 'message', 'is_turn', name)
    return msg_rate


def get_cat_msg_rate(events, levels):
    tf = prep_tf(events)
    for i in range(len(levels)):
        curr_levels = levels[: i+1]
        for turn in range(1, 7):
            tf = tf.join(get_msg_rate(events.copy(), curr_levels, turn))
    return tf


def get_cat_lstg_counts(events, levels):
    # initialize output dataframe
    tf = prep_tf(events)
    # loop over hierarchy, exlcuding lstg
    for i in range(len(levels)):
        curr_levels = levels[: i+1]
        # open listings
        events = events.sort_values([CLOCK] + curr_levels)

        #########
        # REMOVED OPEN LSTGS
        # tfname = '_'.join([curr_levels[-1], 'lstgs_open'])
        # tf[tfname] = open_lstgs(events, curr_levels)
        ##########

        # count features grouped by current level
        ct_feats = events[['lstg_ind', THREAD, 'slr_offer',
                           'byr_offer', ACCEPT]].groupby(by=curr_levels).sum()
        ctl_feats = events[['lstg_ind', THREAD, 'slr_offer',
                            'byr_offer', ACCEPT]].groupby(by=curr_levels + [LSTG]).sum()
        ct_feats = ct_feats - ctl_feats
        per_lstg_feats = [ACCEPT, 'slr_offer', 'byr_offer', THREAD]
        divisor = ct_feats['lstg_ind'].copy()
        divisor[divisor == 0] = 1
        for feat in per_lstg_feats:
            ct_feats[feat] = ct_feats[feat] / divisor

        ct_feats = ct_feats.rename(lambda x: '_'.join([curr_levels[-1], x]) + 's', axis=1)
        tf = tf.join(ct_feats)
        tf.index = tf.index.droplevel(level=curr_levels)

    # collapse to lstg
    tf = tf.rename(lambda curr_name:
                   curr_name if 'lstg_ind' not in curr_name else curr_name.replace('_ind', ''),
                   axis=1)
    return tf.sort_index()


def get_cat_accepts(events, levels):
    # loop over hierarchy, excluding lstg
    events['accept_norm'] = events.price[events[ACCEPT] & ~events.bin] / events[START_PRICE]
    tf = prep_tf(events)

    for i in range(len(levels)):
        curr_levels = levels[: i + 1]
        # quantiles of (normalized) accept price over 30-day window
        quants = fast_quantiles(events, curr_levels, 'accept_norm')
        tf = tf.join(quants)
    # collapse to lstg
    return tf.sort_index()


def get_cat_con(events, levels):
    # loop over hierarchy, exlcuding lstg
    # bin / thread (excluding curr)
    tf = prep_tf(events)

    for i in range(len(levels)):
        curr_levels = levels[: i + 1]
        perc = get_perc(events,  curr_levels, 'bin', 'lstg_ind', name='bin')
        tf = tf.join(perc)

        # quantiles of (normalized) accept price over 30-day window
        quants = fast_quantiles(events, curr_levels, 'first_offer')
        tf = tf.join(quants)

    # collapse to lstg
    return tf.sort_index()


def get_cat_delay(events, levels):
    tf = prep_tf(events)
    clock_df = events.clock.unstack()
    delay = pd.DataFrame(index=clock_df.index)
    delay[0] = np.NaN  # excluding turn 0
    for i in range(1, 8):
        if i in clock_df.columns:
            delay[i] = (clock_df[i] - clock_df[i - 1]).dt.total_seconds()
            if i > 1:
                max_delay = MAX_DELAY_TURN
                censored = delay[i] > max_delay
            else:  # excluding turn 1
                censored = delay.index
                max_delay = 1
            delay.loc[censored, i] = np.NaN
            delay[i] = delay[i] / max_delay
    events['delay'] = delay.rename_axis('index', axis=1).stack()
    # excluding auto rejections
    events.loc[events['delay'] == 0, 'delay'] = np.NaN
    for i in range(len(levels)):
        curr_levels = levels[: i + 1]
        tf = tf.join(fast_quantiles(events, curr_levels, 'slr_delay'))
        tf = tf.join(fast_quantiles(events, curr_levels, 'byr_delay'))
        tf = tf.join(get_expire_perc(events, curr_levels, byr=False))
        tf = tf.join(get_expire_perc(events, curr_levels, byr=True))
    # collapse to lstg
    return tf.sort_index()


def get_cat_start_price(events, levels):
    return get_cat_quantiles_wrapper(events, levels, 'start_price_pctile')


def get_cat_byr_hist(events, levels):
    events.loc[events.base, BYR_HIST] = np.NaN
    events.loc[:, BYR_HIST] = feat_to_pctile(events[BYR_HIST])
    return get_cat_quantiles_wrapper(events, levels, BYR_HIST)


def get_cat_arrival(events, levels):
    return get_cat_quantiles_wrapper(events, levels, 'arrival_rate')


def get_cat_quantiles_wrapper(events, levels, featname):
    tf = prep_tf(events)
    for i in range(len(levels)):
        curr_levels = levels[: i + 1]
        tf = tf.join(fast_quantiles(events, curr_levels, featname))
    return tf.sort_index()


def create_obs(df, is_start, cols):
    toAppend = pd.DataFrame(index=df.index, columns=[INDEX] + cols)
    for c in [ACCEPT, 'message']:
        if c in cols:
            toAppend[c] = False
    if is_start:
        toAppend.loc[:, REJECT] = False
        toAppend.loc[:, INDEX] = 0
        toAppend.loc[:, 'price'] = df[START_PRICE]
        toAppend.loc[:, CLOCK] = df[START_TIME]
    else:
        toAppend.loc[:, REJECT] = True
        toAppend.loc[:, INDEX] = 1
        toAppend.loc[:, 'price'] = np.nan
        toAppend.loc[:, CLOCK] = df[END_TIME]
    return toAppend.set_index(INDEX, append=True)


def expand_index(df, levels):
    df.set_index(levels, append=True, inplace=True)
    idxcols = levels + [LSTG, THREAD]
    if INDEX in df.index.names:
        idxcols += [INDEX]
    df = df.reorder_levels(idxcols)
    df.sort_values(idxcols, inplace=True)
    return df


def add_start_end(offers, listings, levels):
    # listings dataframe
    lstgs = listings[levels + [START_DATE, END_TIME, START_PRICE]].copy()
    lstgs[THREAD] = 0
    lstgs.set_index(THREAD, append=True, inplace=True)
    lstgs = expand_index(lstgs, levels)
    lstgs[START_TIME] = lstgs[START_DATE] * DAY
    lstgs.drop(START_DATE, axis=1, inplace=True)
    lstgs = lstgs.join(offers[ACCEPT].groupby(LSTG).max())
    # create data frames to append to offers
    cols = list(offers.columns)
    start = create_obs(lstgs, True, cols)
    end = create_obs(lstgs[lstgs.accept != 1], False, cols)
    # append to offers
    offers = offers.append(start, sort=True).append(end, sort=True)
    # sort
    return offers.sort_index()


def create_events(data=None, levels=None):
    # initial offers data frame
    offers = data['offers'].join(data['threads'][BYR_HIST])
    offers = offers.join(data['listings'][levels])
    offers = expand_index(offers, levels)
    # add start times and expirations for unsold listings
    events = add_start_end(offers, data['listings'], levels)
    # add features for later use
    events[BYR] = events.index.isin(IDX[BYR], level=INDEX)
    events = events.join(data['listings'][[START_PRICE, 'arrival_rate']])
    events['start_price_pctile'] = feat_to_pctile(data['listings'][START_PRICE])
    return events

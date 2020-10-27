import numpy as np
import pandas as pd
import torch
from collections.abc import Iterable
from utils import unpickle, load_featnames
from rlenv.util import load_featnames
from utils import load_file
from constants import INPUT_DIR, INDEX_DIR, BYR_DROP, \
    SLR, BYR, IDX
from featnames import SLR, X_LSTG, X_OFFER, LOOKUP, NORM, AUTO, \
    CLOCK, LSTG, INDEX, DEC_PRICE, START_PRICE, MSG, THREAD, MODELS, FIRST_ARRIVAL_MODEL


class Subsetter:
    def __init__(self, multi_value=False, level='lstg'):
        self.multi_value = multi_value
        self.index_is_cached = False
        self.multi_index = False
        self.cached_index = None
        self.level = level
        self.index = None

    def set_multi_index(self, feats_df=None):
        if not self.index_is_cached:
            if feats_df is None:
                self.multi_index = False
            else:
                self.multi_index = isinstance(feats_df.index, pd.MultiIndex)

    def cache_index(self, value=None, feats_df=None):
        # make value a list or index
        if not self.multi_value:
            value = [value]
        if feats_df is None:
            self.index = None
        else:
            # extract all values of target level
            if self.multi_index:
                lstgs = feats_df.index.get_level_values(level=self.level)
            else:
                lstgs = feats_df.index
            # check whether the target value appears in the target level
            matches = lstgs.isin(value)
            self.index = lstgs.isin(value) if matches.any() else None
        self.index_is_cached = True

    def __call__(self, feats_df=None):
        if self.index is not None:
            subset = feats_df.loc[self.index, :]
            if self.multi_index and not self.multi_value:
                subset.index = subset.index.droplevel(level=self.level)
        else:
            subset = None
        return subset


def subset_inputs(models=None, input_data=None, value=None, level=None):
    inputs = dict()
    multi_value = isinstance(value, Iterable)
    if models is None:
        models = list(input_data.keys())
    input_data = input_data.copy()
    for model in models:
        inputs[model] = dict()
        subsetter = Subsetter(multi_value=multi_value, level=level)
        for input_group, feats_df in input_data[model].items():
            if not subsetter.index_is_cached:
                subsetter.set_multi_index(feats_df=feats_df)
                subsetter.cache_index(feats_df=feats_df, value=value)
            subset = subsetter(feats_df=feats_df)
            inputs[model][input_group] = subset
    return inputs


def compare_input_dicts(model=None, stored_inputs=None, env_inputs=None):
    for feat_set_name, stored_feats in stored_inputs.items():
        env_feats = env_inputs[feat_set_name]
        feat_eq = torch.lt(torch.abs(torch.add(-stored_feats, env_feats)), 1e-4)
        if not torch.all(feat_eq):
            print('Model input inequality found for {} in {}'.format(model, feat_set_name))
            feat_eq = (~feat_eq.numpy())[0, :]
            feat_eq = np.nonzero(feat_eq)[0]
            featnames = load_featnames(model)
            if 'offer' in feat_set_name:
                featnames = featnames['offer']
            else:
                featnames = featnames[feat_set_name]
            for feat_index in feat_eq:
                print('-- INCONSISTENCY IN {} --'.format(featnames[feat_index]))
                print('stored value = {} | env value = {}'.format(stored_feats[0, feat_index],
                                                                  env_feats[0, feat_index]))
            input("Press Enter to continue...")
            # raise RuntimeError("Environment inputs diverged from true inputs")


def load_model_inputs(model=None, part=None, x_lstg=None):
    index = unpickle(INDEX_DIR + '{}/{}.pkl'.format(part, model))
    featnames = load_featnames(model)
    full_inputs = unpickle(INPUT_DIR + '{}/{}.pkl'.format(part, model))
    x_inputs = full_inputs['x']
    x_idx = pd.Series(index=index, data=full_inputs['idx_x'])
    for k in x_inputs.keys():
        # set column names
        if k == THREAD:
            # use naive values for thread since it's not in featnames
            cols = list(range(x_inputs[k].shape[1]))
        elif 'offer' in k:
            cols = featnames['offer']
        else:
            cols = featnames[k]
        inputs_df = pd.DataFrame(data=x_inputs[k],
                                 index=index,
                                 columns=cols)
        x_inputs[k] = inputs_df
    # fixing x_lstg inputs except 'lstg' b/c it must combine with 'thread'
    x_lstg_sets = list(x_lstg.keys())
    x_lstg_sets.remove(LSTG)
    featnames_list = list(featnames.keys())
    x_lstg_sets = [set_name for set_name in x_lstg_sets
                   if set_name in featnames_list]
    for k in x_lstg_sets:
        vals = x_lstg[k][x_idx, :]
        cols = featnames[k]
        inputs_df = pd.DataFrame(data=vals,
                                 index=x_idx.index,
                                 columns=cols)
        x_inputs[k] = inputs_df

    # fixing lstg
    lstg_vals = x_lstg[LSTG][x_idx, :]
    # drop bo_ct, lstg_ct, & auto decline features from policy byr
    if model == BYR:
        lstg_names = load_featnames(X_LSTG)[LSTG]
        keep = [i for i in range(len(lstg_names))
                if lstg_names[i] not in BYR_DROP]
        lstg_vals = lstg_vals[:, keep]
    thread_vals = x_inputs[THREAD].values
    lstg_vals = np.concatenate((lstg_vals, thread_vals), axis=1)
    x_inputs[LSTG] = pd.DataFrame(
        data=lstg_vals,
        index=x_inputs[THREAD].index,
        columns=featnames[LSTG]
    )
    del x_inputs[THREAD]

    return x_inputs


def load_all_inputs(part=None, byr=False, slr=False):
    x_lstg = load_file(part, X_LSTG)

    models = MODELS.copy()
    models.remove(FIRST_ARRIVAL_MODEL)
    if byr:
        models += [BYR]
    elif slr:
        models += [SLR]

    inputs_dict = dict()
    for model in models:
        inputs_dict[model] = load_model_inputs(model=model,
                                               x_lstg=x_lstg,
                                               part=part)
    return inputs_dict


def reindex_dict(d=None, lstgs=None):
    for m, v in d.items():
        if isinstance(v, dict):
            for k, df in v.items():
                if len(df.index.names) == 1:
                    d[m][k] = df.reindex(index=lstgs)
                else:
                    d[m][k] = df.reindex(index=lstgs, level='lstg')
        else:
            if len(v.index.names) == 1:
                d[m] = v.reindex(index=lstgs)
            else:
                d[m] = v.reindex(index=lstgs, level='lstg')
    return d


def populate_inputs(full_inputs=None, value=None, agent_byr=False, agent=False):
    inputs = dict()
    for feat_set_name, feat_df in full_inputs.items():
        if value is not None:
            curr_set = full_inputs[feat_set_name].loc[value, :].copy()
        else:
            curr_set = full_inputs[feat_set_name].copy()
        # silence messages from agents
        if agent and 'offer' in feat_set_name:
            turn = int(feat_set_name[-1])
            if turn % 2 == int(agent_byr):
                curr_set[MSG] = 0
        curr_set = curr_set.values
        curr_set = torch.from_numpy(curr_set).float()
        if len(curr_set.shape) == 1:
            curr_set = curr_set.unsqueeze(0)
        inputs[feat_set_name] = curr_set
    return inputs


def drop_duplicated_timestamps(part=None, chunk=None):
    # timestemps
    lstgs = chunk[LOOKUP].index
    clock = load_file(part, CLOCK).reindex(index=lstgs, level=LSTG)
    # remove duplicate timestamps within thread
    toDrop = clock.groupby(clock.index.names[:-1]).apply(
        lambda x: x.duplicated())
    clock = clock[~toDrop]
    # flag listings with duplicate timestamps across threads
    flag = clock.groupby(LSTG).apply(lambda x: x.duplicated())
    flag = flag.groupby(LSTG).max()
    # drop flagged listings
    output_lstgs = lstgs.drop(flag[flag].index)
    return output_lstgs


def get_auto_safe_lstgs(chunk=None):
    """
    Drops lstgs where there's at least one offer within 1%
    of the accept or reject price if that price is non-null
    """
    # print('Lstg count: {}'.format(len(lookup)))
    lookup = chunk[LOOKUP].copy()
    # normalize decline prices
    lookup[DEC_PRICE] /= lookup[START_PRICE]
    # drop offers that are 0 or 1 (and very near or 1)
    offers = chunk[X_OFFER].copy()
    near_null_offers = (offers[NORM] >= .99) | (offers[NORM] <= 0.01)
    offers = offers.loc[~near_null_offers, :]

    # extract lstgs that no longer appear in offers
    # these should be kept because they  must have no offers
    # or consist entirely of offers adjacent to null acc/rej prices
    null_offer_lstgs = lookup.index[~lookup.index.isin(
        offers.index.get_level_values(LSTG).unique())]

    # inner join offers with lookup
    offers = offers.join(other=lookup, on=LSTG)
    offers['diff_dec'] = (offers[NORM] - offers[DEC_PRICE]).abs()
    offers['low_diff'] = offers['diff_dec'] < 0.01
    low_diff_count = offers['low_diff'].groupby(level=LSTG).sum()
    no_low_diffs_lstgs = low_diff_count.index[low_diff_count == 0]
    output_lstgs = no_low_diffs_lstgs.union(null_offer_lstgs)
    return output_lstgs


def get_byr_lstgs(chunk=None):
    return chunk[X_OFFER].index.get_level_values(LSTG).unique()


def get_slr_lstgs(chunk=None):
    # keep all lstgs with at least 1 non-auto seller offer
    auto = chunk[X_OFFER][AUTO]
    valid = ~auto[auto.index.isin(IDX[SLR], level=INDEX)]
    s = valid.groupby(LSTG).sum()
    output_lstgs = s[s > 0].index
    return output_lstgs

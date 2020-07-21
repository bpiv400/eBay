import numpy as np
import pandas as pd
from compress_pickle import load
import torch
from collections.abc import Iterable
from utils import load_featnames
from rlenv.util import load_featnames
from utils import load_file
from constants import MODELS, INPUT_DIR, INDEX_DIR, FEATS_DIR,\
    FIRST_ARRIVAL_MODEL, POLICY_MODELS, PARTS_DIR, POLICY_BYR
from featnames import CENSORED, EXP, DELAY, MSG


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


def subset_df(df=None, lstg=None):
    """
    Subsets an arbitrary dataframe to only contain rows for the given
    lstg
    :param df: pd.DataFrame
    :param lstg: integer giving lstg
    :return: pd.Series or pd.DataFrame
    """
    df = df.copy()
    if isinstance(df.index, pd.MultiIndex):
        lstgs = df.index.unique(level='lstg')
        if lstg in lstgs:
            return df.xs(lstg, level='lstg', drop_level=True)
    else:
        if lstg in df.index:
            return df.loc[lstg, :]
    return None


def load_model_inputs(model=None, input_dir=None,
                      index_dir=None, lstgs=None, x_lstg=None):
    input_path = '{}{}.gz'.format(input_dir, model)
    index_path = '{}{}.gz'.format(index_dir, model)
    index = load(index_path)
    featnames = load_featnames(model)
    full_inputs = load(input_path)
    x_inputs = full_inputs['x']
    x_idx = pd.Series(index=index,
                      data=full_inputs['idx_x'])
    contains = None
    for feat_set_name in list(x_inputs.keys()):
        # set column names
        if feat_set_name == 'thread':
            # use naive values for thread since it's not in featnames
            cols = list(range(x_inputs[feat_set_name].shape[1]))
        elif 'offer' in feat_set_name:
            cols = featnames['offer']
        else:
            cols = featnames[feat_set_name]
        inputs_df = pd.DataFrame(data=x_inputs[feat_set_name],
                                 index=index,
                                 columns=cols)
        if contains is None:
            full_lstgs = inputs_df.index.get_level_values('lstg')
            contains = full_lstgs.isin(lstgs)
        inputs_df = inputs_df.loc[contains, :]
        x_inputs[feat_set_name] = inputs_df
    x_idx = x_idx[contains]
    # fixing x_lstg inputs except 'lstg'
    # b/c it must combine with 'thread'
    x_lstg_sets = list(x_lstg.keys())
    x_lstg_sets.remove('lstg')
    featnames_list = list(featnames.keys())
    x_lstg_sets = [set_name for set_name in x_lstg_sets
                   if set_name in featnames_list]
    for feat_set_name in x_lstg_sets:
        vals = x_lstg[feat_set_name][x_idx, :]
        cols = featnames[feat_set_name]
        inputs_df = pd.DataFrame(data=vals,
                                 index=x_idx.index,
                                 columns=cols)
        x_inputs[feat_set_name] = inputs_df

    # fixing lstg
    lstg_vals = x_lstg['lstg'][x_idx, :]
    # drop bo_ct, lstg_ct, & auto acc/dec features from policy byr
    if model == POLICY_BYR:
        first = lstg_vals[:, :4]
        middle = lstg_vals[:, 6:-4]
        lstg_vals = np.concatenate((first, middle), axis=1)
    thread_vals = x_inputs['thread'].values
    cols = featnames['lstg']
    lstg_vals = np.concatenate((lstg_vals, thread_vals), axis=1)
    x_inputs['lstg'] = pd.DataFrame(
        data=lstg_vals,
        index=x_inputs['thread'].index,
        columns=cols
    )
    del x_inputs['thread']

    return x_inputs


def load_all_inputs(part=None, lstgs=None):
    input_dir = '{}{}/'.format(INPUT_DIR, part)
    index_dir = '{}{}/'.format(INDEX_DIR, part)
    x_lstg = load('{}{}/x_lstg.pkl'.format(PARTS_DIR, part))
    inputs_dict = dict()
    models = MODELS + POLICY_MODELS
    models.remove(FIRST_ARRIVAL_MODEL)
    for model in models:
        inputs_dict[model] = load_model_inputs(model=model,
                                               x_lstg=x_lstg,
                                               input_dir=input_dir,
                                               index_dir=index_dir,
                                               lstgs=lstgs)
    return inputs_dict


def load_reindex(part=None, name=None, lstgs=None):
    df = load_file(part, name)
    df = df.reindex(index=lstgs, level='lstg')
    if name == 'x_offer':
        df[CENSORED] = df[EXP] & (df[DELAY] < 1)
    return df


def lstgs_without_duplicated_timestamps(lstgs=None):
    # load timestamps
    offers = load(FEATS_DIR + 'offers.pkl').reindex(index=lstgs, level='lstg')

    # remove censored offers
    clock = offers.loc[~offers.censored, 'clock']

    # remove duplicate timestamps within thread
    toDrop = clock.groupby(['lstg', 'thread']).apply(lambda x: x.duplicated())
    clock = clock[~toDrop]

    # flag listings with duplicate timestamps across threads
    flag = clock.groupby('lstg').apply(lambda x: x.duplicated())
    flag = flag.groupby('lstg').max()

    # drop flagged listgins
    lstgs = lstgs.drop(flag[flag].index)

    return lstgs


def subset_lstgs(df=None, lstgs=None):
    full_lstgs = df.index.get_level_values('lstg')
    contains = full_lstgs.isin(lstgs)
    df = df.loc[contains, :]
    return df


def populate_test_model_inputs(full_inputs=None, value=None, agent_byr=False, agent=False):
    inputs = dict()
    for feat_set_name, feat_df in full_inputs.items():
        if value is not None:
            curr_set = full_inputs[feat_set_name].loc[value, :].copy()
        else:
            curr_set = full_inputs[feat_set_name].copy()
        # silence messages from agent
        if agent and 'offer' in feat_set_name:
            remainder = 1 if agent_byr else 0
            turn = int(feat_set_name[-1])
            if turn % 2 == remainder:
                curr_set[MSG] = 0
        curr_set = curr_set.values
        curr_set = torch.from_numpy(curr_set).float()
        if len(curr_set.shape) == 1:
            curr_set = curr_set.unsqueeze(0)
        inputs[feat_set_name] = curr_set
    return inputs

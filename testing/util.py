import numpy as np
import pandas as pd
import torch
from collections.abc import Iterable
from utils import load_featnames


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
    assert len(stored_inputs) == len(env_inputs)
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

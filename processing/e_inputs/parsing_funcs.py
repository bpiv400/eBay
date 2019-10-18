import sys, pickle
from datetime import datetime as dt
import numpy as np, pandas as pd
import torch

from constants import *
from utils import *


# Converts data frames to tensors sorted (descending) by N_turns.
def convert_to_arrays(d):
    # for feed-forward networks
    if 'x_time' not in d:
        d['x_fixed'] = torch.tensor(d['x_fixed'].reindex(
            d['y'].index).astype(np.float32).values)
        d['y'] = torch.tensor(d['y'].astype(np.float32).values)

    # for recurrent networks
    else:
        # sorting index
        idxnames = d['y'].index.names
        s = d['y'].groupby(idxnames[:-1]).transform('count').rename('count')
        s = s.reset_index().sort_values(['count'] + idxnames,
            ascending=[False] + [True for i in range(len(idxnames))])
        s = s.set_index(idxnames).squeeze()

        # number of turns
        turns = d['y'].groupby(idxnames[:-1]).count()
        turns = turns.sort_values(ascending=False)
        d['turns'] = torch.tensor(turns.values)

        # outcome
        d['y'] = torch.tensor(np.transpose(d['y'].unstack().reindex(
            index=turns.index).astype(np.float32).values))

        # fixed features
        d['x_fixed'] = torch.tensor(d['x_fixed'].reindex(
            index=turns.index).astype(np.float32).values).unsqueeze(dim=0)

        # timestep features
        arrays = []
        for c in d['x_time'].columns:
            array = d['x_time'][c].astype(np.float32).unstack().reindex(
                index=turns.index).values
            arrays.append(np.expand_dims(np.transpose(array), axis=2))
        d['x_time'] = torch.tensor(np.concatenate(arrays, axis=2))

    return d


def add_clock_feats(x_time, model):
    clock = pd.to_datetime(x_time.clock, unit='s', origin=START)
    # US holiday indicator
    x_time['holiday'] = clock.isin(HOLIDAYS)
    # day of week indicator
    for i in range(6):
        x_time['dow' + str(i)] = clock.dt.dayofweek == i
    # minute in day
    if model in ['byr', 'slr']:
        x_time['minutes'] = clock.dt.hour * 60 + clock.dt.minute
    return x_time.drop('clock', axis=1)


def parse_time_feats_delay(model, idx, z_start, z_role):
    # initialize output
    x_time = pd.DataFrame(index=idx).join(z_start)
    # add period
    x_time['period'] = idx.get_level_values('period')
    # time of each pseudo-observation
    x_time['clock'] += INTERVAL[model] * x_time.period
    # features from clock
    x_time = add_clock_feats(x_time, model)
    # time-varying features
    return x_time.join(z_role.reindex(index=idx, fill_value=0))


def parse_fixed_feats_delay(model, x_lstg, x_thread, x_offer):
    # initialize output dataframe
    idx = x_offer.index[x_offer.index.isin(IDX[model], level='index')]
    x_fixed = pd.DataFrame(index=idx)
    # turn indicators
    x_fixed = add_turn_indicators(x_fixed)
    # lstg and byr attributes
    x_fixed = x_fixed.join(x_lstg).join(x_thread)
    # last 2 offers
    drop = [c for c in x_offer.columns if c.endswith('_diff')]
    df = x_offer.drop(drop, axis=1)
    offer1 = df.groupby(['lstg', 'thread']).shift(
        periods=1).reindex(index=idx)
    offer2 = df.groupby(['lstg', 'thread']).shift(
        periods=2).reindex(index=idx)
    x_fixed = x_fixed.join(offer1.rename(
        lambda x: x + '_other', axis=1))
    x_fixed = x_fixed.join(offer2.rename(
        lambda x: x + '_last', axis=1))
    return x_fixed



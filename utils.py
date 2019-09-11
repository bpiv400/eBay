import argparse
import pickle
# from datetime import datetime as dt
import numpy as np
import pandas as pd
import torch

from constants import *


# returns booleans for whether offer is round and ends in nines
def do_rounding(offer):
    digits = np.ceil(np.log10(offer.clip(lower=0.01)))
    factor = 5 * np.power(10, digits-3)
    diff = np.round(offer / factor) * factor - offer
    is_round = diff == 0
    is_nines = (diff > 0) & (diff <= factor / 5)
    return is_round, is_nines


# returns dataframe with US holiday and day-of-week indicators
def extract_day_feats(clock):
    df = pd.DataFrame(index=clock.index)
    df['holiday'] = clock.dt.date.astype('datetime64').isin(HOLIDAYS)
    for i in range(6):
        df['dow' + str(i)] = clock.dt.dayofweek == i
    return df


# Converts data frames to tensors sorted (descending) by N_turns.
def convert_to_tensors(d):
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


# adds indicators for bargaining turn to df, excluding last turn
def add_turn_indicators(df):
    indices = np.unique(df.index.get_level_values('index'))
    for i in range(len(indices)-1):
        ind = indices[i]
        featname = 't' + str((ind+1) // 2)
        df[featname] = df.index.isin([ind], level='index')
    return df


# creates time input features for non-delay byr and slr RNN models
def parse_time_feats_role(model, outcome, x_offer):
    # initialize output dataframe
    idx = x_offer.index[x_offer.index.isin(IDX[model], level='index')]
    x_time = pd.DataFrame(index=idx)
    # current offer
    curr = x_offer.loc[idx]
    # last offer from other role
    offer1 = x_offer.groupby(['lstg', 'thread']).shift(
        periods=1).reindex(index=idx)
    # last offer from same role
    offer2 = x_offer.groupby(['lstg', 'thread']).shift(
        periods=2).reindex(index=idx)
    if model == 'byr':
        start = x_offer.xs(0, level='index')
        start = start.assign(index=1).set_index('index', append=True)
        offer2 = offer2.dropna().append(start).sort_index()
        # remove features that are constant for buyer
        offer2 = offer2.drop(['auto', 'exp', 'reject'], axis=1)
    else:
        offer1 = offer1.drop(['auto', 'exp', 'reject'], axis=1)
    # current offer
    excluded = ['nines', 'auto', 'exp', 'reject']
    if outcome in ['round', 'msg', 'con', 'accept', 'reject']:
        excluded += ['round']
        if outcome in ['msg', 'con', 'accept', 'reject']:
            excluded += ['msg']
            if outcome in ['con', 'accept', 'reject']:
                excluded += ['con', 'norm', 'split']
    last_vars = [c for c in offer2.columns if c in excluded]
    # join dataframes
    x_time = x_time.join(curr.drop(excluded, axis=1))
    x_time = x_time.join(offer1.rename(
        lambda x: x + '_other', axis=1))
    x_time = x_time.join(offer2[last_vars].rename(
        lambda x: x + '_last', axis=1))
    # add turn indicators and return
    return add_turn_indicators(x_time)


def get_clock_feats(time):
    """
    Gets clock features as np.array given the time
    (day of week indicators -- 6 excluded)

    Will need to add argument to include minutes for other models

    :param time: int giving time in seconds
    :return: NA
    """
    out = np.zeros(7, dtype='float32')
    clock = pd.to_datetime(time.clock, unit='s', origin=START)
    if clock.dayofweek < 6:
        out[clock.dayofweek + 1] = 1
    out[0] = clock.isin(HOLIDAYS)
    return out


# helper function to add US holiday, day of week, and minutes in day
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


# creates time input features for byr and slr delay models
def parse_time_feats_delay(model, idx, z):
    # initialize output
    x_time = pd.DataFrame(index=idx).join(z['start'])
    # add period
    x_time['period'] = idx.get_level_values('period')
    # time of each pseudo-observation
    x_time['clock'] += INTERVAL[model] * x_time.period
    # features from clock
    x_time = add_clock_feats(x_time, model)
    # time-varying features
    return x_time.join(z[model].reindex(index=idx, fill_value=0))


# creates fixed input features for non-delay byr and slr models
def parse_fixed_feats_role(x):
    return x['thread'].join(x['lstg'])


# creates fixed input features for byr and slr models
def parse_fixed_feats_delay(model, x):
    # initialize output dataframe
    idx = x['offer'].index[x['offer'].index.isin(
        IDX[model], level='index')]
    x_fixed = pd.DataFrame(index=idx)
    # turn indicators
    x_fixed = add_turn_indicators(x_fixed)
    # lstg and byr attributes
    x_fixed = x_fixed.join(x['lstg']).join(x['thread'])
    # last 2 offers
    drop = [c for c in x['offer'].columns if c.endswith('_diff')]
    df = x['offer'].drop(drop, axis=1)
    offer1 = df.groupby(['lstg', 'thread']).shift(
        periods=1).reindex(index=idx)
    offer2 = df.groupby(['lstg', 'thread']).shift(
        periods=2).reindex(index=idx)
    x_fixed = x_fixed.join(offer1.rename(
        lambda x: x + '_other', axis=1))
    x_fixed = x_fixed.join(offer2.rename(
        lambda x: x + '_last', axis=1))
    return x_fixed


# creates fixed input features for non-days arrival models
def parse_fixed_feats_arrival(outcome, x):
    # thread-level attributes
    threads = x['offer'].xs(1, level='index')
    # intialize output
    x_fixed = pd.DataFrame(index=threads.index).join(x['lstg'])
    # days since lstg start, holiday and day of week
    dow = [v for v in threads.columns if v.startswith('dow')]
    x_fixed = x_fixed.join(threads[['days', 'holiday'] + dow].rename(
        lambda x: 'focal_' + x, axis=1))
    # return or add features
    if outcome == 'loc':
        return x_fixed
    x_fixed = x_fixed.join(x['thread']['byr_us'])
    if outcome == 'hist':
        return x_fixed
    x_fixed = x_fixed.join(x['thread']['byr_hist'])
    if outcome in ['bin', 'sec']:
        return x_fixed

def unpickle(file):
    """
    Unpickles a .pkl file encoded with Python 3
    :param file: str giving path to file
    :return: contents of file
    """
    return pickle.load(open(file, "rb"))

# creates fixed input features for days arrival model
def parse_fixed_feats_days(x, idx):
    # lstg features
    x_fixed = x['lstg'].reindex(index=idx, level='lstg')
    # period
    x_fixed['focal_days'] = x_fixed.index.get_level_values('period')
    # days since lstg start
    day = x_fixed.start_days + x_fixed.focal_days
    clock = pd.to_datetime(day, unit='D', origin=START)
    x_fixed = x_fixed.join(extract_day_feats(clock).rename(
        lambda x: 'focal_' + x, axis=1))
    return x_fixed


# translates command-line experiment id to neural net parameters
def parse_params(args):
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    # returns the path of the appropriate experiments file
    path = 'repo/simulator/experiments/hidden'
    # parameter K for mixture models
    if args['outcome'] in ['sec', 'con']:
        path += '_K'
    try:
        return pd.read_csv(path + '.csv', index_col=0).loc[args['id']]
    except:
        print('No experiment #%d.' % args.id)
        exit()
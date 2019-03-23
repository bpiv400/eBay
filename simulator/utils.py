import argparse
import string
import sys
import pickle
import math
from datetime import datetime as dt
import numpy as np
import pandas as pd
import torch
import torch.nn.utils.rnn as rnn
import torch.autograd.gradcheck as gradcheck
from constants import *


def ln_beta_pdf(a, b, z):
    '''
    Computes the log density of the beta distribution with parameters [a, b].

    Inputs:
        - a (3, N, K): alpha parameters
        - b (3, N, K): beta parameters
        - z (3, N): observed outcome (i.e., delay or concession)

    Output:
        - lnf (3, N, K): log density under beta distribution
    '''
    z = z.unsqueeze(dim=2)
    lbeta = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
    la = torch.mul(a-1, torch.log(z))
    lb = torch.mul(b-1, torch.log(1-z))
    return la + lb - lbeta


def dlnLda(a, b, z, gamma, indices):
    '''
    Computes the derivative of the log of the beta pdf with respect to alpha.

    Inputs:
        - a (3, N, K): alpha parameters
        - b (3, N, K): beta parameters
        - z (3, N): observed outcome (i.e., delay or concession)
        - gamma (3, N, K): mixture weights
        - indices (3, N): 1.0 when 0 < z < 1

    Output:
        - da (3, N, K): derivative of lnL wrt alpha
    '''
    z = z.unsqueeze(dim=2)
    da = torch.log(z) - torch.digamma(a) + torch.digamma(a + b)
    da *= indices.unsqueeze(dim=2) * gamma
    da[torch.isnan(da)] = 0
    return da


def get_rounded(offer):
    digits = np.ceil(np.log10(offer.clip(lower=0.01)))
    factor = 5 * np.power(10, digits-3)
    rounded = np.round(offer / factor) * factor
    is_round = rounded == offer
    is_nines = ((rounded > offer) & (rounded - offer <= factor / 5))
    return rounded, is_round, is_nines


def reshape_wide(df, threads):
    dfs = [df.xs(t, level='turn').loc[threads] for t in range(1, 4)]
    arrays = [dfs[t].values.astype(float) for t in range(3)]
    reshaped = [np.reshape(arrays[t], (1, len(threads), -1)) for t in range(3)]
    ttuple = tuple([torch.tensor(reshaped[i]) for i in range(3)])
    return torch.cat(ttuple, 0).float()


def convert_to_tensors(data):
    '''
    Converts data frames to tensors sorted (descending) by N_turns.
    '''
    tensors = {}
    # order threads by sequence length
    turns = MAX_TURNS - data['y'].isna().groupby(level='thread').sum()
    turns = turns.sort_values(ascending=False)
    threads = turns.index
    tensors['turns'] = torch.tensor(turns.values).int()

    # outcome
    tensors['y'] = torch.squeeze(reshape_wide(data['y'], threads))

    # categorical fixed features
    tensors['x_cat'] = {}
    for col in data['x_cat'].columns:
        s = data['x_cat'][col].loc[threads].values
        tensors['x_cat'][col] = torch.tensor(s).unsqueeze(dim=0)

    # non-categorical fixed features
    M_fixed = data['x_fixed'].loc[threads].astype(np.float64).values
    tensors['x_fixed'] = torch.tensor(M_fixed).float().unsqueeze(dim=0)

    # offer features
    tensors['x_offer'] = reshape_wide(data['x_offer'], threads)

    return tensors


def add_turn_feats(x_offer):
    '''
    Creates indicator for each turn
    '''
    # small dataframe of turns
    turns = pd.DataFrame(0, index=[1, 2, 3], columns=['t1', 't2', 't3'])
    turns.index.name = 'turn'
    turns.loc[1, 't1'] = 1
    turns.loc[2, 't2'] = 1
    turns.loc[3, 't3'] = 1
    # join with offer features
    return x_offer.join(turns, on='turn')


def add_time_feats(x, pre):
    days = x[pre + '_days']
    if pre == 's_prev':
        x[pre + '_auto'] = days == 0
        x[pre + '_exp'] = days == 2
    time = x[pre + '_time']
    x[pre + '_year'] = time.dt.year - 2012
    x[pre + '_week'] = time.dt.week
    x[pre + '_minutes'] = 60 * time.dt.hour + time.dt.minute
    dow = time.dt.dayofweek
    for i in range(7):
        x[pre + '_dow' + str(i)] = dow == i
    return x


def add_offer_feats(x, pre, rounded):
    con = x[pre + '_con']
    x[pre + '_rej'] = con == 0
    x[pre + '_half'] = np.abs(con - 0.5) < TOL_HALF
    x[pre + '_round'] = rounded
    x[pre + '_lnround'] = np.log(1+rounded)
    return x


def expand_offer_feats(x, pre, model):
    # delay and time variables
    if pre != 's_curr' or model in MODELS[1:]:
        x = add_time_feats(x, pre)
    # concession and rounded offer
    rounded, is_round, is_nines = get_rounded(x[pre + '_offer'])
    if pre != 's_curr' or model in MODELS[2:]:
        x = add_offer_feats(x, pre, rounded)
    # boolean for round offer
    if pre != 's_curr' or model in MODELS[3:]:
        x[pre + '_isround'] = is_round
    # boolean for nines
    if pre != 's_curr' or model in MODELS[4:]:
        x[pre + '_isnines'] = is_nines
    # drop columns
    x.drop([pre + s for s in ['_time', '_offer']], axis=1, inplace=True)
    if pre == 's_curr':
        if model in MODELS[:2]:
            x.drop('s_curr_con', axis=1, inplace=True)
        if model in MODELS[0]:
            x.drop('s_curr_days', axis=1, inplace=True)
    return x.reindex(sorted(x.columns), axis=1)


def expand_x_offer(x, model):
    '''
    A wrapper function that loops over recent offers.
    '''
    for pre in ['s_prev', 'b', 's_curr']:
        x = expand_offer_feats(x, pre, model)
    return add_turn_feats(x)


def get_x_fixed(T):
    '''
    Constructs a dataframe of fixed features that are used to initialize the
    hidden state and the LSTM cell.
    '''
    # initialize dataframe
    x = pd.DataFrame(index=T.index)
    # decline and accept price
    for z in ['decline', 'accept']:
        x[z] = T[z + '_price']
        x['ln' + z] = np.log(1 + x[z])
        x[z + '_round'], x[z + '_isround'], x[z +
                                              '_isnines'] = get_rounded(x[z])
        x['has_' + z] = x[z] > 0. if z == 'decline' else x[z] < T['start_price']
    # binary features
    for z in BINARY_FEATS:
        x[z] = T[z]
    # count variables
    for z in COUNT_FEATS:
        x[z] = T[z]
        x['ln' + z] = np.log(1 + x[z])
        x['has_' + z] = T[z] > 0
    # feedback score
    x['fdbk_pstv'] = T['fdbk_pstv']
    x['fdbk_100'] = T['fdbk_pstv'] == 100.
    # shipping speed
    for z in ['ship_fast', 'ship_slow']:
        x['has_' + z] = (T[z] != -1.).astype(np.bool) & ~T[z].isna()
    return x.reindex(sorted(x.columns), axis=1)


def get_batch_indices(count, mbsize):
    '''
    Creates matrix of randomly sampled minibatch indices.
    '''
    v = [i for i in range(count)]
    np.random.shuffle(v)
    batches = int(np.ceil(count / mbsize))
    indices = [sorted(v[mbsize*i:mbsize*(i+1)]) for i in range(batches)]
    return indices


def initialize_gamma(N, K):
    '''
    Creates a (3, N_obs, K) tensor filled with 1/K.
    '''
    if K == 0:
        return None
    gamma = np.full(tuple(N) + (K,), 1/K)
    return torch.as_tensor(gamma, dtype=torch.float)


def get_gamma(a, b, y):
    '''
    Calculates gamma from the predicted beta densities.
    '''
    dens = torch.exp(ln_beta_pdf(a, b, y))
    return torch.div(dens, torch.sum(dens, dim=2, keepdim=True))


def get_lnL(simulator, d):
    x_offer = rnn.pack_padded_sequence(d['x_offer'], d['turns'])
    p, a, b = simulator(d['x_fixed'], x_offer)
    gamma = get_gamma(a, b, d['y'])
    criterion = simulator.get_criterion()
    lnL = -criterion(p, a, b, d['y'], gamma, simulator).item()
    return lnL / torch.sum(d['turns']).item(), p, a, b, gamma


def check_gradient(simulator, d):
    '''
    Implements the Torch gradcheck function to numerically verify
    analytical gradients.
    '''
    x_fixed = d['x_fixed'][:, :N_SAMPLES, :]
    x_offer = d['x_offer'][:, :N_SAMPLES, :]
    turns = d['turns'][:N_SAMPLES]
    y = d['y'][:, :N_SAMPLES].double()
    p, a, b = simulator(x_fixed, rnn.pack_padded_sequence(x_offer, turns))
    criterion = simulator.get_criterion()
    if simulator.get_model() in ['delay', 'con']:
        gamma = initialize_gamma(y.size(), simulator.get_K()).double()
        gradcheck(criterion, (p.double(), a.double(), b.double(), y, gamma))
    else:
        gradcheck(criterion, (p.double(), y))


def get_args():
    '''
    Command-line parser for following arguments:
        --gradcheck: boolean for verifying gradients
        --model: the outcome to predict, one of:
            ['delay', 'con', 'round', 'nines', 'msg']
    '''
    parser = argparse.ArgumentParser(
        description='Model of environment for bargaining AI.')
    parser.add_argument('--id', default=1, type=int,
                        help='Experiment ID.')
    parser.add_argument('--gradcheck', default=False, type=bool,
                        help='Boolean flag to first check gradients.')
    parser.add_argument('--model', default='delay', type=str,
                        help='One of: delay, con, round, nines, msg.')
    return parser.parse_args()

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


def reshape_wide(df, threads):
    dfs = [df.xs(t, level='turn').loc[threads] for t in range(1, 4)]
    arrays = [dfs[t].values.astype(float) for t in range(3)]
    reshaped = [np.reshape(arrays[t], (1, len(threads), -1)) for t in range(3)]
    ttuple = tuple([torch.tensor(reshaped[i]) for i in range(3)])
    return torch.cat(ttuple, 0).float()


def convert_to_tensors(d):
    '''
    Converts data frames to tensors sorted (descending) by N_turns.
    '''
    tensors = {}
    # order threads by sequence length
    turns = MAX_TURNS - d['y'].isna().groupby(level=['lstg', 'thread']).sum()
    turns = turns.sort_values(ascending=False)
    threads = turns.index
    tensors['turns'] = torch.tensor(turns.values).int()

    # outcome
    tensors['y'] = reshape_wide(d['y'], threads).squeeze()

    # non-categorical fixed features
    M_fixed = d['x_fixed'].loc[threads].astype(np.float64).values
    tensors['x_fixed'] = torch.tensor(M_fixed).float().unsqueeze(dim=0)

    # offer features
    tensors['x_offer'] = reshape_wide(d['x_offer'], threads)

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


def get_x_fixed(T, lda_weights, slr, model):
    '''
    Constructs a dataframe of fixed features that are used to initialize the
    hidden state and the LSTM cell.
    '''
    # initialize dataframe
    x = pd.DataFrame(index=T.index)
    # start date
    date = pd.to_datetime(T['start_date'], unit='D', origin=ORIGIN)
    x['year'] = date.dt.year - 2012
    x['week'] = date.dt.week
    for i in range(7):
        x['dow' + str(i)] = date.dt.dayofweek == i
    # prices
    for z in ['start', 'decline', 'accept']:
        x[z] = T[z + '_price']
        if z != 'start' or model == 'cat':
            x[z + '_round'], x[z +'_nines'] = do_rounding(x[z])
    x['has_decline'] = x['decline'] > 0.
    x['has_accept'] = x['accept'] <  T['start_price']
    # features without transformations
    for z in ['fdbk_pstv'] + BINARY_FEATS + COUNT_FEATS:
        x[z] = T[z]
    # leaf LDA scores
    w = lda_weights[:,T.leaf]
    for i in range(len(lda_weights)):
        x['lda' + str(i)] = w[i, :]
    # one-hot vectors
    for z in ['meta', 'cndtn']:
        for i in range(0, np.max(T[z])+1):
            x[z + str(i)] = T[z] == i
    # indicator for perfect feedback score
    x['fdbk_100'] = x['fdbk_pstv'] == 100.
    # shipping speed
    for z in ['ship_fast', 'ship_slow']:
        x['has_' + z] = (T[z] != -1.).astype(np.bool) & ~T[z].isna()
    # initial time-valued features
    if model != 'delay':
        for key, val in slr.items():
            if key not in ['days', 'offer', 'norm', 'con', 'msg']:
                x[key] = val[1]
    return x


def reshape_long(df, name=None):
    col = df.columns
    if col[0] != 1:
        df = df.rename(columns={col[0]: 1, col[1]: 2, col[2]: 3})
    s = df.stack(dropna=False)
    s.index.rename(['lstg', 'thread', 'turn'], inplace=True)
    if name is not None:
        s.rename(name, inplace=True)
    return s


def do_rounding(offer):
    digits = np.ceil(np.log10(offer.clip(lower=0.01)))
    factor = 5 * np.power(10, digits-3)
    diff = np.round(offer / factor) * factor - offer
    is_round = (diff == 0).rename('round')
    is_nines = ((diff > 0) & (diff <= factor / 5)).rename('nines')
    return is_round, is_nines


def add_y(x, d, pre, feats=[], islast=True, include=True):
    col = [1, 2, 3] if islast else [2, 3, 4]
    if not include:
        feats = [k for k in d.keys() if k not in feats + ['offer', 'con']]
    for key, val in d.items():
        if key in feats:
            x['_'.join([pre, key])] = reshape_long(val[col])
    return x


def add_y_binary(x, d, pre, islast):
    col = [1, 2, 3] if islast else [2, 3, 4]
    if pre == 's':
        x['s_auto'] = x['s_days'] == 0  # auto-accept/reject
        x['s_exp'] = x['s_days'] == 2   # expired
    # round and nines indicators
    offer = reshape_long(d['offer'][col])
    x[pre + '_round'], x[pre + '_nines'] = do_rounding(offer)
    # reject and split-the-difference indicators
    con = reshape_long(d['con'][col])
    x[pre + '_rej'] = con == 0
    x[pre + '_half'] = np.abs(con - 0.5) < TOL_HALF
    return x


def get_x_offer(byr, slr, model):
    """
    Creates a df with simulator input at each turn.
    Index: [lstg, thread, turn] (turn in range 1..3)
    """
    x = pd.DataFrame()
    # last buyer offer
    x = add_y(x, byr, 'b', include=False)
    x = add_y_binary(x, byr, 'b', True)
    # slr offer
    if model == 'delay':
        x = add_y(x, slr, 's', include=False)
    elif model == 'con':
        x = add_y(x, slr, 's', feats=['norm', 'msg'])
        x = add_y(x, slr, 's', feats=['norm', 'msg'],
            islast=False, include=False)
    elif model == 'cat':
        x = add_y(x, slr, 's', feats=['msg'])
        x = add_y(x, slr, 's', feats=['msg'], islast=False, include=False)
    x = add_y_binary(x, slr, 's', model != 'cat')
    return add_turn_feats(x)


def trim_threads(s):
    '''
    Restricts series to threads with at least one non-NA value.
    '''
    return s[s.isna().groupby(level=['lstg','thread']).sum() < 3]


def get_y(d, model):
    '''
    Creates a series of model-specific outcome.
    '''
    # delay
    delay = reshape_long(d['days'] / 2, name='delay')
    delay.loc[delay == 0] = np.nan    # automatic responses
    if model == 'delay':
        return trim_threads(delay)
    # concession
    con = reshape_long(d['con'], name='con')
    con.loc[delay.isna()] = np.nan
    con.loc[delay == 1] = np.nan    # expired offers
    if model == 'con':
        return trim_threads(con)
    # categorical outcome
    msg = reshape_long(d['msg'], name='msg')
    offer = reshape_long(d['offer'], name='offer')
    is_round, is_nines = do_rounding(offer)
    cat = is_round + 2 * is_nines + 3 * msg
    cat.loc[con.isna()] = np.nan
    cat.loc[con == 1] = np.nan    # accepts
    cat.loc[con == 0] = np.nan    # rejects
    return trim_threads(cat)


def split_byr_slr(df):
    '''
    Splits a dataframe with turn indices for columns into separate
    dataframes corresponding to byr and solr turns.
    '''
    b = df[[1, 3, 5, 7]].rename(columns={3: 2, 5: 3, 7: 4})
    s = df[[0, 2, 4, 6]].rename(columns={0: 1, 4: 3, 6: 4})
    return b, s


def get_days(clock):
    '''
    Creates two series, one for the byr and one for the slr, counting
    the number of days (as a float) since the last offer.
    '''
    # initialize dataframes
    b_days = pd.DataFrame(index=clock.index, columns=[1, 2, 3, 4])
    s_days = pd.DataFrame(index=clock.index, columns=[1, 2, 3, 4])
    # first delay
    s_days[1] = 0.
    # remaining delays
    for i in range(1, 8):
        sec = clock[i] - clock[i-1]
        if i % 2:
            b_days[int((i+1)/2)] = sec / (24 * 3600)
        else:
            sec[sec > 48 * 3600] = 48 * 3600    # temporary fix
            s_days[int(1 + i/2)] = sec / (24 * 3600)
    assert np.min(b_days.stack()) > 0
    assert np.min(s_days.stack()) >= 0
    return b_days, s_days


def get_con(offers):
    '''
    Creates dataframes of concessions at each turn,
    separately for buyer and seller.
    '''
    # initialize dataframes
    b_con = pd.DataFrame(index=offers.index, columns=[1, 2, 3, 4])
    s_con = pd.DataFrame(index=offers.index, columns=[1, 2, 3, 4])
    # first concession
    b_con[1] = offers[1] / offers[0]
    s_con[1] = 0
    assert np.count_nonzero(np.isnan(b_con[1].values)) == 0
    # remaining concessions
    for i in range(2, 8):
        norm = (offers[i] - offers[i-2]) / (offers[i-1] - offers[i-2])
        if i % 2:
            b_con[int((i+1)/2)] = norm
        else:
            s_con[int(1 + i/2)] = norm
    # verify that all concessions are in bounds
    assert np.nanmax(b_con.values) <= 1 and np.nanmin(b_con.values) >= 0
    assert np.nanmax(s_con.values) <= 1 and np.nanmin(s_con.values) >= 0
    return b_con, s_con


def create_variables(O, T, time_feats):
    '''
    Creates dictionaries of variables by role.
    '''
    # unstack offers and msgs by index
    offers = O['price'].unstack()
    offers[0] = T['start_price']
    msgs = O['message'].unstack()
    msgs[0] = 0

    # seconds since beginning of lstg
    clock = time_feats['clock'].drop(0, level='thread').unstack()
    clock = clock.join(time_feats['clock'].groupby('lstg').min().rename(0))
    clock = clock.loc[msgs.index]

    # dictionary of time-valued features
    df = time_feats.drop('clock', axis=1)
    df0 = df.xs(0, level='thread').xs(0, level='index')
    df = df.drop(0, level='thread')
    d_time = {c: df0[c].rename(0).to_frame().join(
        df[c].unstack()).loc[msgs.index] for c in df.columns}

    # dictionaries by role
    byr = {}
    slr = {}
    byr['days'], slr['days'] = get_days(clock)
    byr['offer'], slr['offer'] = split_byr_slr(offers)
    byr['norm'], slr['norm'] = split_byr_slr(offers.div(offers[0], axis=0))
    byr['con'], slr['con'] = get_con(offers)
    byr['msg'], slr['msg'] = split_byr_slr(msgs)
    for key, val in d_time.items():
        byr[key], slr[key] = split_byr_slr(val)

    return byr, slr


def load_data():
    # training data
    data = pickle.load(open(BASEDIR + 'input/train.pkl', 'rb'))
    O, T, time_feats = [data[key] for key in ['O', 'T', 'time_feats']]

    # LDA weights from slr-leaf matrix
    lda_weights = pickle.load(open(LDADIR + 'weights.pkl', 'rb'))

    # fill in NA values
    T.loc[T['fdbk_pstv'].isna(), 'fdbk_pstv'] = 100
    T.loc[T['fdbk_score'].isna(), 'fdbk_score'] = 0
    T.loc[T['slr_hist'].isna(), 'slr_hist'] = 0

    return O, T, time_feats, lda_weights


def process_inputs(model):
    # load dataframes
    print('Loading data')
    O, T, time_feats, lda_weights = load_data()

    # split into role-specific dictionaries of dataframes
    print('Creating variables')
    byr, slr = create_variables(O, T, time_feats)

    # dictionary of tensors
    d = {}
    d['y'] = get_y({key: val[[2, 3, 4]] for key, val in slr.items()}, model)
    d['x_fixed'] = get_x_fixed(T, lda_weights, slr, model)
    d['x_offer'] = get_x_offer(byr, slr, model)
    return convert_to_tensors(d)


def prepare_batch(train, g, idx):
    '''
    Slices data and component weights, and puts them in dictionary.
    '''
    batch = {}
    batch['y'] = train['y'][:, idx]
    batch['x_fixed'] = train['x_fixed'][:, idx, :]
    batch['x_offer'] = rnn.pack_padded_sequence(
        train['x_offer'][:, idx, :], train['turns'][idx])
    if g is not None:
        batch['g'] = g[:, idx, :]
    return batch


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
    return parser.parse_args()

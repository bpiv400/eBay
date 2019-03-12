import argparse, string
import numpy as np, pandas as pd
import torch, torch.nn.utils.rnn as rnn
import torch.autograd.gradcheck as gradcheck

MAX_TURNS = 3
N_SAMPLES = 100
CON_PRE = tuple('s_curr_' + x for x in ['con', 'rej', 'digit', 'offer', 'lnoffer'])
TOL_HALF = 0.02 # count concessions within this range as 1/2


def ln_beta_pdf(a, b, z):
    lbeta = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
    la = torch.mul(a-1, torch.log(z))
    lb = torch.mul(b-1, torch.log(1-z))
    return la + lb - lbeta


def dlnLda(a, b, z, indices):
    da = indices * (torch.log(z) - torch.digamma(a) + torch.digamma(a + b))
    da[torch.isnan(da)] = 0
    return da


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
    # order threads by sequence length
    turns = MAX_TURNS - data['y'].isna().groupby(level='thread').sum()
    turns = turns.sort_values(ascending=False)
    threads = turns.index
    turns = torch.tensor(turns.values).int()

    # outcome
    y = torch.squeeze(reshape_wide(data['y'], threads))

    # fixed features
    M_fixed = data['x_fixed'].loc[threads].astype(np.float64).values
    x_fixed = torch.tensor(np.reshape(M_fixed, (1, len(threads), -1))).float()

    # offer features
    x_offer = reshape_wide(data['x_offer'], threads)

    return x_offer, x_fixed, y, turns


def add_offer_feats(x_offer):
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


def get_digit_indicators(offer, digit, prefix):
    '''
    Creates a dataframe with indicators for the following:
        a. offer == digit in hundredths
        b. offer == digit in tenths
        c. offer == digit in ones
        d. offer == digit in tens
        e. offer == digit in hundreds
    '''
    strings = 'XXXXX' + offer.map('{0:.2f}'.format)
    df = pd.DataFrame(index=strings.index)
    indices = [-1, -2, -4, -5, -6]
    for i in range(len(indices)):
        # define place
        newname = '_'.join([prefix, 'digit', string.ascii_lowercase[i] + digit])
        df[newname] = strings.str[indices[i]] == digit
    return df


def expand_offer_vars(x, prefix, model):
    # delay and time variables
    if model != 'delay' or prefix != 's_curr':
        days = x[prefix + '_days']
        if prefix == 's_prev':
            x = x.join((days == 0).rename('_'.join([prefix, 'auto'])))
            x = x.join((days == 2).rename('_'.join([prefix, 'exp'])))
        time = x[prefix + '_time']
        year = time.dt.year - 2012
        for i in range(2):
            x = x.join((year == i).rename('_'.join([prefix, 'year' + str(i)])))
        dow = time.dt.dayofweek
        for i in range(7):
            x = x.join((dow == i).rename('_'.join([prefix, 'dow' + str(i)])))
        week = time.dt.week
        for i in range(1, 53):
            x = x.join((week == i).rename('_'.join([prefix, 'week' + str(i)])))
        hour = time.dt.hour
        for i in range(24):
            x = x.join((hour == i).rename('_'.join([prefix, 'hour' + str(i)])))
    x.drop(prefix + '_time', axis=1, inplace=True)
    # concession and offer variables
    if model == 'msg' or prefix != 's_curr':
        con = x[prefix + '_con']
        x = x.join((con == 0).rename('_'.join([prefix, 'rej'])))
        x = x.join((np.abs(con - 0.5) < TOL_HALF).rename('_'.join([prefix, 'half'])))
        offer = x[prefix + '_offer']
        x = x.join(get_digit_indicators(offer, '9', prefix))
        x = x.join(get_digit_indicators(offer, '0', prefix))
        x = x.join(np.log(1 + offer).rename('_'.join([prefix, 'lnoffer'])))
    if prefix == 's_curr':
        if model != 'msg':
            x.drop(['s_curr_offer', 's_curr_con'], axis=1, inplace=True)
        if model == 'delay':
            x.drop('s_curr_days', axis=1, inplace=True)
    return x


def expand_x_offer(x, model):
    for prefix in ['s_prev', 'b', 's_curr']:
        x = expand_offer_vars(x, prefix, model)
    return x

def get_x_fixed(T):
    # initialize dataframe
    x = pd.DataFrame(index=T.index)
    # decline and accept price
    for z in ['decline', 'accept']:
        x[z] = T[z + '_price']
        x['ln' + z] = np.log(1 + x[z])
        x['has_' + z] = x[z] > 0. if z == 'decline' else x[z] < T['start_price']
        x = x.join(get_digit_indicators(x[z], '9', z))
        x = x.join(get_digit_indicators(x[z], '0', z))
    return x


def get_batch_indices(count, mbsize):
    # create matrix of randomly sampled minibatch indices
    v = [i for i in range(count)]
    np.random.shuffle(v)
    batches = int(np.ceil(count / mbsize))
    indices = [sorted(v[mbsize*i:mbsize*(i+1)]) for i in range(batches)]
    return batches, indices


def compute_example(simulator):
    x_fixed = torch.tensor([[[20.]]])
    x_offer = torch.tensor([[[0., 0., 0.5, 1., 0., 0.]],
                            [[0., 0., 0.5, 0., 1., 0.]],
                            [[0.5, 0., 0.5, 0., 0., 1.]]])
    x_offer = rnn.pack_padded_sequence(x_offer, torch.tensor([3]))
    p, a, b = simulator(x_fixed, x_offer)
    print('\tOffer of $10 on $20 listing:')
    print('\t\tp_expire = %.2f' % p[0,:,0].item())
    print('\t\tp_reject = %.2f' % p[0,:,1].item())
    print('\t\tp_accept = %.2f' % p[0,:,2].item())
    print('\t\tp_50 = %.2f' % p[0,:,3].item())
    print('\t\tp_beta = %.2f' % (1-torch.sum(p[0,:,:])).item())
    print('\t\talpha = %2.1f' % a[0,:].item())
    print('\t\tbeta = %2.1f' % b[0,:].item())


def check_gradient(simulator, criterion, x_fixed, x_offer, y, turns):
    x_fixed = x_fixed[:, :N_SAMPLES, :]
    x_offer = x_offer[:, :N_SAMPLES, :]
    turns = turns[:N_SAMPLES]
    y = y[:, :N_SAMPLES].double()
    theta = simulator(x_fixed, rnn.pack_padded_sequence(x_offer, turns))
    gradcheck(criterion, (theta.double(), y))


def get_args():
    parser = argparse.ArgumentParser(
        description='Model of environment for bargaining AI.')
    parser.add_argument('--id', default=1, type=int,
        help='Experiment ID.')
    parser.add_argument('--model', default='con', type=str,
        help='Model outcome: delay, con, or msg.')
    parser.add_argument('--gradcheck', default=False, type=bool,
        help='Boolean flag to first check gradients.')
    return parser.parse_args()
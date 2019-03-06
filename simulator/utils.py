import argparse
import numpy as np, pandas as pd
import torch, torch.nn.utils.rnn as rnn
import torch.autograd.gradcheck as gradcheck

MAX_TURNS = 3
N_SAMPLES = 100


def convert_to_tensors(data):
    '''
    Converts data frames to tensors sorted (descending) by N_turns.
    '''

    # order threads by sequence length
    turns = 3 - pd.isna(data['y']).sum(1)
    turns = turns.sort_values(ascending=False)
    threads = turns.index
    turns = torch.tensor(turns.values)

    # seller concessions
    M_y = np.transpose(data['y'].loc[threads].values)
    y = torch.tensor(np.reshape(M_y, (3, len(threads), 1))).float()

    # fixed features
    M_fixed = data['x_fixed'].loc[threads].values
    x_fixed = torch.tensor(np.reshape(M_fixed, (1, len(threads), -1))).float()

    # offer features
    dfs = [data['x_offer'].xs(t, level='turn').loc[threads] for t in range(1, 4)]
    arrays = [dfs[t].values.astype(float) for t in range(3)]
    reshaped = [np.reshape(arrays[t], (1, len(threads), -1)) for t in range(3)]
    ttuple = tuple([torch.tensor(reshaped[i]) for i in range(3)])
    x_offer = torch.cat(ttuple, 0).float()

    return x_offer, x_fixed, y, turns


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
    print('\t\tp_reject = %.2f' % p[0,:,0].item())
    print('\t\tp_accept = %.2f' % p[0,:,1].item())
    print('\t\tp_50 = %.2f' % p[0,:,2].item())
    print('\t\tp_beta = %.2f' % (1-torch.sum(p[0,:,:])).item())
    print('\t\talpha = %2.1f' % a[0,:].item())
    print('\t\tbeta = %2.1f' % b[0,:].item())


def check_gradient(simulator, criterion, x_fixed, x_offer, y, turns):
    x_fixed = x_fixed[:, :N_SAMPLES, :]
    x_offer = x_offer[:, :N_SAMPLES, :]
    turns = turns[:N_SAMPLES]
    y = y[:, :N_SAMPLES, :]
    p, a, b = simulator(x_fixed, rnn.pack_padded_sequence(x_offer, turns))
    gradcheck(criterion, (p.double(), a.double(), b.double(), y.double()))


def get_args():
    parser = argparse.ArgumentParser(
        description='Model of environment for bargaining AI.')
    parser.add_argument('--mbsize', default=128, type=int,
        help='Number of samples per minibatch.')
    parser.add_argument('--epochs', default=50, type=float,
        help='Number of runs through the data.')
    parser.add_argument('--hidden', default=100, type=int,
        help='Number of nodes in each hidden layer.')
    parser.add_argument('--dropout', default=0.5, type=float,
        help='Dropout rate.')
    parser.add_argument('--step_size', default=0.001, type=float,
        help='Learning rate for optimizer.')
    parser.add_argument('--layers', default=2, type=int,
        help='Number of recurrent layers.')
    parser.add_argument('--is_lstm', default=True, type=bool,
        help='Boolean flag to use an LSTM rather than an RNN.')
    parser.add_argument('--check_grad', default=False, type=bool,
        help='Boolean flag to first check gradients.')
    return parser.parse_args()
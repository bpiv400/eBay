import numpy as np, pandas as pd
import torch, torch.nn.utils.rnn as rnn
import torch.autograd.gradcheck as gradcheck

MAX_TURNS = 3
N_SAMPLES = 100

def get_batch_indices(count, mbsize):
    # create matrix of randomly sampled minibatch indices
    v = [i for i in range(count)]
    np.random.shuffle(v)
    batches = int(np.ceil(count / mbsize))
    indices = [sorted(v[mbsize*i:mbsize*(i+1)]) for i in range(batches)]
    return batches, indices


def compute_example(simulator):
    x_fixed = torch.tensor([[[20.]]])
    x_offer = torch.tensor([[[0., 0.5, 1., 0., 0.]],
                            [[0., 0.5, 0., 1., 0.]],
                            [[0.5, 0.5, 0., 0., 1.]]])
    x_offer = rnn.pack_padded_sequence(x_offer, torch.tensor([3]))
    p, a, b = simulator(x_fixed, x_offer)
    print(p)
    print(a)
    print(b)


def check_gradient(simulator, criterion, x_fixed, x_offer, y, turns):
    x_fixed = x_fixed[:, :N_SAMPLES, :]
    x_offer = x_offer[:, :N_SAMPLES, :]
    turns = turns[:N_SAMPLES]
    y = y[:, :N_SAMPLES, :]
    p, a, b = simulator(x_fixed, rnn.pack_padded_sequence(x_offer, turns))
    gradcheck(criterion, (p.double(), a.double(), b.double(), y.double()))
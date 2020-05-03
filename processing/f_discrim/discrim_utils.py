import numpy as np
import torch
from torch.nn.functional import log_softmax
from compress_pickle import dump
from processing.e_inputs.inputs_utils import save_sizes, \
    convert_x_to_numpy, save_small
from utils import load_model
from train.train_consts import MBSIZE
from constants import TRAIN_RL, VALIDATION, INPUT_DIR


def save_discrim_files(part, name, x_obs, x_sim):
    """
    Packages discriminator inputs for training.
    :param part: string name of partition.
    :param name: string name of model.
    :param x_obs: dictionary of observed data.
    :param x_sim: dictionary of simulated data.
    :return: None
    """
    # featnames and sizes
    if part == VALIDATION:
        save_sizes(x_obs, name)

    # indices
    idx_obs = x_obs['lstg'].index
    idx_sim = x_sim['lstg'].index

    # create dictionary of numpy arrays
    x_obs = convert_x_to_numpy(x_obs, idx_obs)
    x_sim = convert_x_to_numpy(x_sim, idx_sim)

    # y=1 for real data
    y_obs = np.ones(len(idx_obs), dtype=bool)
    y_sim = np.zeros(len(idx_sim), dtype=bool)
    d = {'y': np.concatenate((y_obs, y_sim), axis=0)}

    # join input variables
    assert x_obs.keys() == x_sim.keys()
    d['x'] = {k: np.concatenate((x_obs[k], x_sim[k]), axis=0) for k in x_obs.keys()}

    # save inputs
    dump(d, INPUT_DIR + '{}/{}.gz'.format(part, name))

    # save small
    if part == TRAIN_RL:
        save_small(d, name)


def get_model_predictions(m, x):
    # initialize neural net
    net = load_model(m, verbose=False)
    if torch.cuda.is_available():
        net = net.to('cuda')

    # split into batches
    v = np.array(range(len(x['lstg'])))
    batches = np.array_split(v, 1 + len(v) // MBSIZE[False])

    # model predictions
    p0 = []
    for b in batches:
        x_b = {k: torch.from_numpy(v[b, :]) for k, v in x.items()}
        if torch.cuda.is_available():
            x_b = {k: v.to('cuda') for k, v in x_b.items()}
        theta_b = net(x_b).cpu().double()
        p0.append(np.exp(log_softmax(theta_b, dim=-1)))

    # concatenate and return
    return torch.cat(p0, dim=0).numpy()
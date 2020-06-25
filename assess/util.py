import numpy as np
import torch
from torch.nn.functional import log_softmax, nll_loss
from train.Sample import get_batches
from utils import load_model, load_file
from constants import TEST


def load_threads_offers(sim=False):
    suffix = '_sim' if sim else ''
    threads = load_file(TEST, 'x_thread{}'.format(suffix))
    offers = load_file(TEST, 'x_offer{}'.format(suffix))
    return threads, offers


def get_model_predictions(name, data):
    """
    Returns predictions from model
    :param name: string name of model.
    :param data: corresponding EBayDataset.
    :return: (p, lnL)
        - p: Nxk array of probabilities.
        - lnL: N-length vector of log-likelihoods.
    """
    # initialize neural net
    net = load_model(name, verbose=False)
    if torch.cuda.is_available():
        net = net.to('cuda')

    # get predictions from neural net
    lnp, lnl = [], []
    batches = get_batches(data)
    for b in batches:
        if torch.cuda.is_available():
            b['x'] = {k: v.to('cuda') for k, v in b['x'].items()}
        theta = net(b['x']).cpu().double()
        if theta.size()[1] == 1:
            theta = torch.cat((torch.zeros_like(theta), theta), dim=1)
        lnp.append(log_softmax(theta, dim=-1))
        lnl.append(-nll_loss(lnp[-1], b['y'], reduction='none'))

    # concatenate and convert to numpy
    lnp = torch.cat(lnp).numpy()
    lnl = torch.cat(lnl).numpy()

    return np.exp(lnp), lnl

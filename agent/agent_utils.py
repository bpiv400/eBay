import math
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softmax
from constants import FIRST_ARRIVAL_MODEL
from utils import load_state_dict, load_model
from agent.agent_consts import FULL_CON, QUARTILES, HALF, AGENT_STATE, NO_SALE_CUTOFF
from rlenv.env_utils import proper_squeeze
from rlenv.Composer import Composer


def get_con_set(con):
    if con == FULL_CON:
        return np.linspace(0, 100, 101) / 100
    elif con == QUARTILES:
        return np.array([0, 0.25, 0.50, 0.75, 1.0])
    elif con == HALF:
        return np.array([0.0, 0.50, 1.0])
    else:
        raise RuntimeError("Invalid concession set type parameter")


def detect_norm(init_dict=None):
    found_v, found_g = False, False
    for param_name in init_dict.keys():
        if '_v' in param_name:
            found_v = True
        elif '_g' in param_name:
            found_g = True
    if found_g and found_v:
        return "weight"
    else:
        raise NotImplementedError("Unexpected normalization type")


def load_agent_params(model=None, run_dir=None):
    """

    :param torch.nn.Module model:
    :param string run_dir:
    :return:
    """
    params = torch.load('{}params.pkl'.format(run_dir), map_location=torch.device('cpu'))
    state_dict = params[AGENT_STATE]
    model.load_state_dict(state_dict=state_dict, strict=True)
    for param in model.parameters(recurse=True):
        param.requires_grad = False
    model.eval()


def align_x_lstg_lookup(x_lstg, lookup):
    x_lstg = pd.concat([df.reindex(index=lookup.index) for df in x_lstg.values()],
                       axis=1)
    return x_lstg


def get_batch_unlikely(x_lstg_chunk=None, model=None):
    for key, val in x_lstg_chunk:
        x_lstg_chunk[key] = torch.from_numpy(x_lstg_chunk[key]).float()
    logits = model(x_lstg_chunk)
    pi = softmax(logits, dim=logits.dim() - 1)
    pi = pi[:, 0]
    unlikely = torch.nonzero(pi > NO_SALE_CUTOFF)
    return unlikely


def process_unlikely(unlikely=None, start_idx=None):
    unlikely + start_idx
    unlikely = proper_squeeze(unlikely)
    unlikely = unlikely.tolist()
    return unlikely


def chunk_x_lstg(i=None, batch_size=None, composer=None, x_lstg=None):
    start_idx = i * batch_size
    end_idx = min(start_idx + batch_size, len(x_lstg.index))
    x_lstg_chunk = x_lstg.iloc[start_idx:end_idx]
    x_lstg_chunk = composer.decompose_x_lstg(x_lstg=x_lstg_chunk)
    return x_lstg_chunk


def remove_unlikely_arrival_lstgs(x_lstg=None, lookup=None):
    """
    Removes listings from x_lstg and lookup DataFrames
    that fall under some minimum threshold for likelihood
    of arrival in the first year

    :param pd.DataFrame x_lstg:
    :param pd.DataFrame lookup:
    :return:
    """
    # error checking
    assert x_lstg.index.name == lookup.index.name
    assert lookup.index.name == 'lstg'
    assert lookup.index.equals(x_lstg.index)
    org_length = len(lookup.index)

    # setup search
    batch_size = 1024.0
    model = load_model(FIRST_ARRIVAL_MODEL)
    composer = Composer(x_lstg.columns)
    chunks = math.ceil(len(x_lstg.index) / batch_size)
    unlikely_lstgs = list()
    # iterate over chunks
    for i in range(chunks):
        x_lstg_chunk = chunk_x_lstg(i=i, batch_size=batch_size,
                                    composer=composer, x_lstg=x_lstg)
        unlikely = get_batch_unlikely(x_lstg_chunk=x_lstg_chunk, model=model)
        if unlikely.numel() > 0:
            unlikely = process_unlikely(unlikely=unlikely, start_idx=i * batch_size)
            unlikely_lstgs = unlikely_lstgs + unlikely

    if len(unlikely_lstgs) > 0:
        unlikely_lstgs = x_lstg.index.values[unlikely_lstgs]
        x_lstg = x_lstg.drop(index=unlikely_lstgs)
        lookup = lookup.drop(index=unlikely_lstgs)
    print('Dropped: {}% of listings'.format(len(unlikely_lstgs) / org_length))
    return x_lstg, lookup


def load_init_model(name=None, size=None):
    state_dict = load_state_dict(name=name)

    # fix output layer size
    output_w_name = 'output.weight'
    output_b_name = 'output.bias'
    org_w = state_dict[output_w_name]
    org_b = state_dict[output_b_name]
    if size == org_b.shape[0]:
        return state_dict
    new_w = torch.zeros((size, org_w.shape[1]), dtype=org_w.dtype,
                        requires_grad=False)
    new_b = torch.zeros(size, dtype=org_b.dtype,
                        requires_grad=False)

    # map rejection onto rejection
    new_w[0, :] = org_w[0, :]
    new_b[0] = org_b[0]

    # map acceptance onto acceptance
    new_w[new_w.shape[0]-1, :] = org_w[org_w.shape[0]-1, :]
    new_b[new_b.shape[0]-1] = org_b[org_b.shape[0]-1]

    # group intermediate values
    num_int_values = size - 2
    org_values_per_int_value = int((org_b.shape[0] - 2) / num_int_values)
    start_idx = 1
    for i in range(1, size - 1):
        stop_idx = (start_idx + org_values_per_int_value)
        new_w[i, :] = torch.sum(org_w[start_idx:stop_idx, :], dim=0)
        new_b[i] = torch.sum(org_b[start_idx:stop_idx])
        start_idx = stop_idx
    state_dict[output_w_name] = new_w
    state_dict[output_b_name] = new_b
    return state_dict

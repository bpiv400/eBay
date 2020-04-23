import os
from compress_pickle import load, dump
import numpy as np
import pandas as pd
import torch
from datetime import datetime as dt
from utils import load_state_dict
from agent.agent_consts import FULL_CON, QUARTILES, HALF, \
    AGENT_STATE, PARAM_DICTS
from constants import RL_LOG_DIR


def get_con_set(con):
    if con == FULL_CON:
        return nmp.linspace(0, 100, 101) / 100
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


def gen_run_id():
    return dt.now().strftime('%y%m%d-%H%M')


def create_params_file(path):
    # append parameters
    series = list()
    for d in PARAM_DICTS:
        for k, v in d.items():
            series.append(pd.Series(name=k, dtype=v['type']))

    # columns for timings
    series.append(pd.Series(name='time_elapsed', dtype=int))
    series.append(pd.Series(name='iterations', dtype=int))

    # create and save dataframe
    df = pd.concat(series, axis=1)
    dump(df, path)


def save_params(role=None,
                run_id=None,
                agent_params=None,
                batch_params=None,
                ppo_params=None,
                time_elapsed=None,
                iterations=None):
    path = RL_LOG_DIR + '{}/runs.pkl'.format(role)

    # if file does not exist, create it
    if not os.path.isfile(path):
        create_params_file(path)

    # open file
    df = load(path)

    # add parameters
    for d in [agent_params, batch_params, ppo_params]:
        for k, v in d.items():
            df.loc[run_id, k] = v

    # add timings
    df.loc[run_id, 'time_elapsed'] = time_elapsed
    df.loc[run_id, 'iterations'] = iterations

    # save file
    dump(df, path)

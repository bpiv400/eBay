import os
from datetime import datetime as dt
from compress_pickle import load, dump
import numpy as np
import pandas as pd
import torch
from rlpyt.samplers.parallel.worker import initialize_worker
from rlpyt.utils.collections import AttrDict
from rlpyt.utils.seed import set_envs_seeds
from utils import load_state_dict
from constants import (RL_LOG_DIR, SLR_VALUE_INIT, SLR_POLICY_INIT,
                       BYR_DELAY_POLICY_INIT, BYR_DELAY_VALUE_INIT,
                       REINFORCE_DIR, SLR_DELAY_VALUE_INIT,
                       SLR_DELAY_POLICY_INIT)
from agent.agent_consts import FULL_CON, QUARTILES, HALF, PARAM_DICTS


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


def load_agent_model(model=None, model_path=None):
    """
    Loads state dict of trained agent.
    :param torch.nn.Module model: agent model.
    :param str model_path: path to agent state dict.
    :return:
    """
    state_dict = torch.load(model_path,
                            map_location=torch.device('cpu'))
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


def create_params_file():
    # append parameters
    series = list()
    for param_set, d in PARAM_DICTS.items():
        if param_set != 'system_params':
            for k, v in d.items():
                dtype = bool if 'type' not in v else v['type']
                series.append(pd.Series(name=k, dtype=dtype))

    # columns for timings
    series.append(pd.Series(name='time_elapsed', dtype=int))

    # create and return dataframe
    df = pd.concat(series, axis=1)
    return df


def save_params(run_id=None,
                args=None,
                time_elapsed=None):
    path = RL_LOG_DIR + '{}/runs.pkl'.format(args['role'])

    # if file does not exist, create it
    if not os.path.isfile(path):
        df = create_params_file()

    # otherwise, open file
    else:
        df = load(path)

    # add parameters
    for k, v in args.items():
        if k in df.columns:
            df.loc[run_id, k] = v

    # add timings
    df.loc[run_id, 'time_elapsed'] = time_elapsed

    # save file
    dump(df, path)


def get_agent_name(byr=False, policy=False, delay=False):
    if byr and delay:
        if policy:
            return BYR_DELAY_POLICY_INIT
        else:
            return BYR_DELAY_VALUE_INIT
    elif not byr and delay:
        if policy:
            return SLR_DELAY_POLICY_INIT
        else:
            return SLR_DELAY_VALUE_INIT
    elif not byr and not delay:
        if policy:
            return SLR_POLICY_INIT
        else:
            return SLR_VALUE_INIT
    else:
        raise NotImplementedError("BYR MUST HAVE DELAY")


def get_train_file_path(rank):
    return '{}train/{}.hdf5'.format(REINFORCE_DIR, rank)


def ebay_sampling_process(common_kwargs, worker_kwargs):
    """Target function used for forking parallel worker processes in the
    samplers. After ``initialize_worker()``, it creates the specified number
    of environment instances and gives them to the collector when
    instantiating it.  It then calls collector startup methods for
    environments and agent.  If applicable, instantiates evaluation
    environment instances and evaluation collector.
    Then enters infinite loop, waiting for signals from master to collect
    training samples or else run evaluation, until signaled to exit.
    """
    c, w = AttrDict(**common_kwargs), AttrDict(**worker_kwargs)
    initialize_worker(w.rank, w.seed, w.cpus, c.torch_threads)
    envs = list()
    for env_rank in w.env_ranks:
        filename = get_train_file_path(env_rank)
        envs.append(c.EnvCls(**c.env_kwargs, filename=filename))
    set_envs_seeds(envs, w.seed)

    collector = c.CollectorCls(
        rank=w.rank,
        envs=envs,
        samples_np=w.samples_np,
        batch_T=c.batch_T,
        TrajInfoCls=c.TrajInfoCls,
        agent=c.get("agent", None),  # Optional depending on parallel setup.
        sync=w.get("sync", None),
        step_buffer_np=w.get("step_buffer_np", None),
        global_B=c.get("global_B", 1),
        env_ranks=w.get("env_ranks", None),
    )
    agent_inputs, traj_infos = collector.start_envs(c.max_decorrelation_steps)
    collector.start_agent()


    ctrl = c.ctrl
    ctrl.barrier_out.wait()
    while True:
        collector.reset_if_needed(agent_inputs)  # Outside barrier?
        ctrl.barrier_in.wait()
        if ctrl.quit.value:
            break
        agent_inputs, traj_infos, completed_infos = collector.collect_batch(
            agent_inputs, traj_infos, ctrl.itr.value)
        for info in completed_infos:
            c.traj_infos_queue.put(info)
        ctrl.barrier_out.wait()

    for env in envs:
        env.close()


def network_size(net):
    bits = 32
    total_bits = 0.0
    for m in net.modules():
        for p in m.parameters():
            s = np.array(p.size())
            b = np.prod(np.array(s)) * bits
            total_bits += b
    return total_bits / 8e6

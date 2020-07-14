import torch
from rlpyt.samplers.parallel.worker import initialize_worker
from rlpyt.utils.collections import AttrDict
from rlpyt.utils.seed import set_envs_seeds
from utils import load_state_dict


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


def compose_args(arg_dict=None, parser=None):
    for k, v in arg_dict.items():
        parser.add_argument('--{}'.format(k), **v)


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
        envs.append(c.EnvCls(**c.env_kwargs, rank=env_rank))
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

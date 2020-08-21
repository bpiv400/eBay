import os
import torch
from agent.const import OPTIM_STATE, AGENT_STATE
from agent.models.AgentModel import AgentModel
from constants import AGENT_DIR, POLICY_SLR, POLICY_BYR
from featnames import SLR, BYR


def get_agent_name(byr=False):
    return POLICY_BYR if byr else POLICY_SLR


def get_log_dir(byr=None):
    role = BYR if byr else SLR
    log_dir = AGENT_DIR + '{}/'.format(role)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    return log_dir


def get_paths(byr=None, name=None):
    # log directory
    log_dir = get_log_dir(byr)

    # run id
    run_id = name

    # concatenate
    run_dir = log_dir + 'run_{}/'.format(run_id)

    return log_dir, run_id, run_dir


def load_agent_model(model_args=None, run_dir=None):
    model = AgentModel(**model_args)
    path = run_dir + 'params.pkl'
    d = torch.load(path, map_location=torch.device('cpu'))
    if OPTIM_STATE in d:
        d = d[AGENT_STATE]
        torch.save(d, path)
    model.load_state_dict(d, strict=True)
    for param in model.parameters(recurse=True):
        param.requires_grad = False
    model.eval()
    return model

import os
from agent.const import ENTROPY_BONUS
from constants import AGENT_DIR, BYR, SLR, POLICY_SLR, POLICY_BYR


def get_agent_name(byr=False):
    return POLICY_BYR if byr else POLICY_SLR


def get_log_dir(byr=None):
    role = BYR if byr else SLR
    log_dir = AGENT_DIR + '{}/'.format(role)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    return log_dir


def get_run_suffix(delta=None, beta=None):
    return 'delta_{}_beta_{}'.format(delta, beta)


def get_run_id(kl=None, delta=None, beta=None):
    suffix = get_run_suffix(delta=delta, beta=beta)
    if kl is None:
        run_id = 'entropy_{}_{}'.format(ENTROPY_BONUS, suffix)
    else:
        run_id = 'kl_{}_{}'.format(kl, suffix)
    return run_id

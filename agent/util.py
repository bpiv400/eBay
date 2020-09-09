import os
import numpy as np
from agent.const import FULL, SPARSE, NOCON
from assess.util import get_sale_norm
from constants import AGENT_DIR, POLICY_SLR, POLICY_BYR, VALIDATION, IDX
from featnames import SLR, BYR, DROPOUT, NORM, AUTO, INDEX, LSTG
from utils import unpickle


def get_agent_name(byr=False):
    return POLICY_BYR if byr else POLICY_SLR


def get_log_dir(byr=None):
    role = BYR if byr else SLR
    log_dir = AGENT_DIR + '{}/'.format(role)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    return log_dir


def get_paths(**kwargs):
    # log directory
    log_dir = get_log_dir(kwargs[BYR])

    # run id
    dropout = [int(10 * kwargs[DROPOUT][i]) for i in range(2)]
    run_id = '{}_{}_{}'.format(kwargs['con_set'], dropout[0], dropout[1])

    # concatenate
    run_dir = log_dir + 'run_{}/'.format(run_id)

    return log_dir, run_id, run_dir


def define_con_set(con_set=None, byr=False):
    if con_set == FULL:
        num_con = 101
    elif con_set == SPARSE:
        num_con = 11
    elif con_set == NOCON:
        num_con = 2
    else:
        raise ValueError('Invalid concession set: {}'.format(con_set))

    cons = np.arange(num_con) / (num_con - 1)
    if not byr:
        cons = np.concatenate([cons, [1.1]])  # expiration rejection
    return cons


def find_best_run(byr=None, sales=False):
    log_dir = get_log_dir(byr=byr)
    df = unpickle(log_dir + 'runs.pkl')
    s = df.xs(VALIDATION, level='part')[NORM]
    s = s.xs('sales' if sales else 'all', level='listings')
    run_id = s[~s.isna()].astype('float64').idxmax()
    print('Best run: {}'.format(run_id))
    return log_dir + '{}/'.format(run_id)


def get_valid_slr(data=None, lookup=None):
    # count non-automatic seller offers
    auto = data['offers'][AUTO]
    valid = ~auto[auto.index.isin(IDX[SLR], level=INDEX)]
    s = valid.groupby(LSTG).sum()
    lstgs = s[s > 0].index.intersection(lookup.index, sort=None)  # listings to keep
    for k, v in data.items():
        data[k] = v.reindex(index=lstgs, level=LSTG)
    lookup = lookup.reindex(index=lstgs)
    return data, lookup


def get_value_slr(offers=None, start_price=None):
    norm = get_sale_norm(offers)
    norm = norm.reindex(index=start_price.index, fill_value=0.)
    price = norm * start_price
    return norm.mean(), price.mean()

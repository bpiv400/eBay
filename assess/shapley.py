import argparse
import numpy as np
import pandas as pd
import torch
import shap
from agent.models.AgentModel import load_agent_model
from agent.util import find_best_run
from utils import unpickle, load_file, load_sizes, load_featnames, get_role
from agent.const import DELTA_CHOICES
from constants import TEST, IDX
from featnames import BYR, SLR, X_LSTG, BYR_HIST, DAYS, DELAY, \
    DAYS_SINCE_LSTG, LSTG, THREAD, CLOCK_FEATS, OUTCOME_FEATS, \
    TIME_FEATS, THREAD_COUNT

BACKGROUND_SIZE = 100
SAMPLES = 10


def collapse_by_group(df=None, byr=None, turn=None):
    # sum by feature group
    cols = dict()
    cols['w2v'] = ['{}{}'.format(name, num)
                   for name in [BYR, SLR]
                   for num in range(32)]
    cols['cndtn'] = ['new', 'used', 'refurb', 'wear']
    cols['slr'] = ['store', 'us', 'fdbk_score', 'fdbk_pstv', 'fdbk_100']
    if not byr:
        cols['slr'] += ['lstg_ct', 'bo_ct']
    cols['price'] = ['start_price_pctile', 'start_is_round', 'start_is_nines']
    if not byr:
        cols['price'] += ['auto_decline', 'has_decline']
    cols['lstg'] = ['fast', 'photos', 'has_photos']
    cols['clock'] = ['{}_{}'.format(feat, i)
                     for feat in CLOCK_FEATS
                     for i in range(1, turn + 1)]
    cols['clock'] += ['start_dow{}'.format(d) for d in range(6)]
    cols['clock'] += ['start_years', 'start_holiday']
    cols['byr'] = [BYR_HIST]
    if not byr:
        cols['tf'] = ['{}_{}'.format(feat, i)
                      for feat in TIME_FEATS
                      for i in range(1, turn + 1)]
        cols['tf'] += [THREAD_COUNT]
    cols['delay'] = ['{}_{}'.format(feat, i)
                     for feat in [DAYS, DELAY]
                     for i in range(1, turn + 1)]
    cols['delay'] += [DAYS_SINCE_LSTG]
    if turn > 1:
        cols['offer'] = ['{}_{}'.format(feat, i)
                         for feat in OUTCOME_FEATS
                         for i in range(1, turn)
                         if feat not in [DAYS, DELAY]]

    groups = pd.DataFrame(index=df.index)
    for k, v in cols.items():
        groups[k] = df[v].sum(axis=1)
    return groups


def wrapper(model=None, x=None):
    return np.argmax(get_pdf(model, x), axis=1) == 0


def get_pdf(model=None, x=None):
    return model(torch.from_numpy(x).float()).numpy()


def compute_shapley_vals(x=None, role=None, model=None):
    idx_rand = np.random.choice(range(np.shape(x)[0]),
                                size=BACKGROUND_SIZE + SAMPLES,
                                replace=False)
    background = x[idx_rand[:BACKGROUND_SIZE], :]
    e = shap.KernelExplainer(lambda z: wrapper(model, z), background)

    idx = idx_rand[BACKGROUND_SIZE:]
    shap_vals = e.shap_values(x[idx, :])

    # put in dataframe
    featnames, names = load_featnames(role), []
    for k, v in featnames.items():
        if k == 'offer':
            for i in range(1, max(IDX[role]) + 1):
                offer_names = ['{}_{}'.format(n, i) for n in v]
                names += offer_names
        else:
            names += v
    df = pd.DataFrame(shap_vals, columns=names)

    return df, idx


def load_inputs_model(delta=None, turn=None):
    # preliminaries
    byr = turn in IDX[BYR]
    run_dir = find_best_run(byr=byr, delta=delta)
    role = get_role(byr)
    # inputs
    d = unpickle(run_dir + '{}/{}.pkl'.format(TEST, role))
    sizes = load_sizes(role)['x']
    x_lstg = load_file(TEST, X_LSTG)
    x = {k: v[d['idx_x'], :] for k, v in x_lstg.items() if k in sizes}
    x[LSTG] = np.concatenate([x[LSTG], d['x'][THREAD]], axis=1)
    for k, v in d['x'].items():
        if k != THREAD:
            x[k] = v
    x = np.concatenate([x[k] for k in sizes.keys()], axis=1)
    # turn number
    if byr:
        dummies = d['x'][THREAD][:, -3:]
        x_turn = 7 - 6 * dummies[:, 0] - 4 * dummies[:, 1] - 2 * dummies[:, 2]
    else:
        dummies = d['x'][THREAD][:, -2:]
        x_turn = 6 - 4 * dummies[:, 0] - 2 * dummies[:, 1]
    # subset to turn
    x = x[x_turn == turn, :]
    # model
    model_args = dict(byr=byr, value=False)
    model = load_agent_model(model_args=model_args, run_dir=run_dir)
    return x, model


def main():
    # parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--turn', type=int, choices=range(1, 8))
    parser.add_argument('--delta', type=float, choices=DELTA_CHOICES)
    args = parser.parse_args()
    byr = args.turn in IDX[BYR]
    role = get_role(byr)

    # components for shapley value estimation
    x, model = load_inputs_model(turn=args.turn, delta=args.delta)

    # shapley values by group
    df, idx = compute_shapley_vals(x=x, role=role, model=model)
    groups = collapse_by_group(df, byr=byr, turn=args.turn)

    # add in P(reject)
    reject = get_pdf(model, x[idx, :])[:, 0] > .99
    s = pd.Series(reject, name='reject')
    groups = pd.concat([s, groups], axis=1)
    print(groups)


if __name__ == '__main__':
    main()

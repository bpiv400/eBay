import argparse
import numpy as np
import pandas as pd
import torch
import shap
from agent.models.AgentModel import load_agent_model
from agent.util import find_best_run, get_turn
from utils import unpickle, load_file, load_sizes, load_featnames, \
    run_func_on_chunks, get_role, topickle
from assess.const import DELTA_SLR, DELTA_BYR
from constants import IDX, BYR_DROP, NUM_CHUNKS, PLOT_DIR
from featnames import BYR, SLR, X_LSTG, BYR_HIST, DAYS, DELAY, \
    DAYS_SINCE_LSTG, LSTG, THREAD, CLOCK_FEATS, OUTCOME_FEATS, \
    TIME_FEATS, THREAD_COUNT, TEST

BACKGROUND_SIZE = 100
SAMPLES = 10
BASE_IDX = {1: 0, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 0}


def collapse_by_group(df=None, turn=None):
    byr = turn in IDX[BYR]

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


def create_df(shap_vals=None, turn=None):
    role = get_role(turn in IDX[BYR])
    featnames, names = load_featnames(role), []
    for k, v in featnames.items():
        if k == 'offer':
            for i in range(1, max(IDX[role]) + 1):
                offer_names = ['{}_{}'.format(n, i) for n in v]
                names += offer_names
        else:
            names += v
    df = pd.DataFrame(shap_vals, columns=names)
    return df


def wrapper(model=None, x=None, base_idx=None):
    return np.argmax(get_pdf(model, x), axis=1) == base_idx


def get_pdf(model=None, x=None):
    return model(torch.from_numpy(x).float()).numpy()


def process_chunk(chunk=None, x=None, model=None, turn=None):
    # subset and shuffle observations
    idx = np.arange(len(x))
    idx = idx[idx % NUM_CHUNKS == chunk]
    np.random.shuffle(idx)
    background = x[idx[:BACKGROUND_SIZE], :]
    test = x[idx[BACKGROUND_SIZE:BACKGROUND_SIZE+SAMPLES], :]

    # shap values from kernel explainer
    e = shap.KernelExplainer(
        lambda z: wrapper(model, z, BASE_IDX[turn]), background)
    print('Base rate in background: {}'.format(e.expected_value))
    shap_vals = e.shap_values(test)

    # dataframe, collapsed by group
    df = create_df(shap_vals=shap_vals, turn=turn)
    groups = collapse_by_group(df, turn=turn)

    return groups


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
    if byr:
        x_lstg_feats = load_featnames(X_LSTG)[LSTG]
        idx_keep = [i for i in range(len(x_lstg_feats))
                    if x_lstg_feats[i] not in BYR_DROP]
        x[LSTG] = x[LSTG][:, idx_keep]
    x[LSTG] = np.concatenate([x[LSTG], d['x'][THREAD]], axis=1)
    for k, v in d['x'].items():
        if k != THREAD:
            assert np.shape(v)[1] == sizes[k]
            x[k] = v
    x = np.concatenate([x[k] for k in sizes.keys()], axis=1)
    # subset to turn
    x = x[get_turn(d['x'][THREAD], byr=byr) == turn, :]
    # model
    model_args = dict(byr=byr, value=False)
    model = load_agent_model(model_args=model_args, run_dir=run_dir)
    return x, model


def main():
    # parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--turn', type=int, choices=range(1, 8))
    turn = parser.parse_args().turn
    delta = DELTA_BYR if turn in IDX[BYR] else DELTA_SLR

    # components for shapley value estimation
    x, model = load_inputs_model(turn=turn, delta=delta)

    groups = run_func_on_chunks(
        f=process_chunk,
        func_kwargs=dict(x=x, model=model, turn=turn)
    )
    groups = pd.concat(groups)

    topickle(groups, PLOT_DIR + 'shap{}.pkl'.format(turn))


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('forkserver')
    main()

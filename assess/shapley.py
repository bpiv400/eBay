from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import shap
from utils import load_inputs, load_file, load_sizes, load_featnames, \
    run_func_on_chunks, get_role, topickle, load_model
from constants import IDX, NUM_CHUNKS, PLOT_DIR
from featnames import BYR, SLR, X_LSTG, TEST, VALUES_MODEL

BACKGROUND_SIZE = 100
SAMPLES = 2


def create_groups(shap_vals=None):
    featnames, names = load_featnames(VALUES_MODEL), []
    for k, v in featnames.items():
        names += v
    df = pd.DataFrame(shap_vals, columns=names)

    # sum by feature group
    cols = dict()
    cols['w2v'] = ['{}{}'.format(name, num)
                   for name in [BYR, SLR]
                   for num in range(32)]
    cols['cndtn'] = ['new', 'used', 'refurb', 'wear']
    cols['slr'] = ['store', 'us', 'lstg_ct', 'bo_ct',
                   'fdbk_score', 'fdbk_pstv', 'fdbk_100']
    cols['price'] = ['start_price_pctile', 'start_is_round', 'start_is_nines']
    cols['auto'] = ['auto_decline', 'has_decline']
    cols['shipping'] = ['fast']
    cols['photos'] = ['photos', 'has_photos']
    cols['date'] = ['start_dow{}'.format(d) for d in range(6)]
    cols['date'] += ['start_years', 'start_holiday']

    assert np.sum([len(v) for v in cols.values()]) == len(df.columns)
    groups = pd.DataFrame(index=df.index)
    for k, v in cols.items():
        groups[k] = df[v].sum(axis=1)

    return groups


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def wrapper(model=None, x=None, sizes=None):
    # convert to tensor
    x = torch.from_numpy(x).float()

    # dictionary of tensors
    d, ct = OrderedDict(), 0
    for k, v in sizes.items():
        d[k] = x[:, ct:ct + v]
        ct += v
    assert x.size()[-1] == ct
    x = d

    # predicted value
    theta = sigmoid(model(x).numpy())

    return theta


def process_chunk(chunk=None, x=None, model=None, sizes=None):
    # subset and shuffle observations
    idx = np.arange(len(x))
    idx = idx[idx % NUM_CHUNKS == chunk]
    np.random.shuffle(idx)
    background = x[idx[:BACKGROUND_SIZE], :]
    test = x[idx[BACKGROUND_SIZE:BACKGROUND_SIZE+SAMPLES], :]

    # shap values from kernel explainer
    e = shap.KernelExplainer(lambda z: wrapper(model, z, sizes=sizes), background)
    print('Average predicted value in background: {}'.format(e.expected_value))
    shap_vals = e.shap_values(test)[0]

    # dataframe, collapsed by group
    groups = create_groups(shap_vals=shap_vals)

    return groups


def main():
    # inputs
    d = load_inputs(TEST, VALUES_MODEL)
    sizes = load_sizes(VALUES_MODEL)['x']
    x_lstg = load_file(TEST, X_LSTG)
    x = {k: v[d['idx_x'], :] for k, v in x_lstg.items() if k in sizes}
    x = np.concatenate([x[k] for k in sizes.keys()], axis=1)

    # model
    model = load_model(VALUES_MODEL)

    # calculate shapley values in parallel
    groups = run_func_on_chunks(
        f=process_chunk,
        func_kwargs=dict(x=x, model=model, sizes=sizes)
    )
    groups = pd.concat(groups)
    print(groups)
    print(np.abs(groups).mean())

    topickle(groups, PLOT_DIR + 'shap.pkl')


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('forkserver')
    main()

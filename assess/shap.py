import numpy as np
import pandas as pd
import torch
import shap
from agent.models.AgentModel import load_agent_model
from agent.util import get_paths
from utils import unpickle, load_file, load_sizes, load_featnames
from agent.const import FULL, NOCON
from constants import TEST, POLICY_SLR
from featnames import BYR, SLR, CON_SET, DELTA, DROPOUT, ENTROPY, X_LSTG, \
    LSTG, THREAD, CLOCK_FEATS, OUTCOME_FEATS, TIME_FEATS, THREAD_COUNT, \
    BYR_HIST, DAYS, DELAY, DAYS_SINCE_LSTG, TURN_FEATS


def wrapper(model=None, x=None):
    pdf = model(torch.from_numpy(x).float()).numpy()
    return np.argmax(pdf, axis=1) == 1


# run parameters
args = {BYR: False,
        DELTA: .75,
        CON_SET: NOCON,
        DROPOUT: (.0, .1),
        ENTROPY: .025}

# run folder
log_dir, run_id, run_dir = get_paths(**args)

# create model
model_args = {k: v for k, v in args.items() if k in [BYR, CON_SET]}
model_args['value'] = False
model = load_agent_model(model_args=model_args, run_dir=run_dir)

# create input tensor
sizes = load_sizes(POLICY_SLR)['x']
d = unpickle(run_dir + '{}/{}.pkl'.format(TEST, POLICY_SLR))
x_lstg = load_file(TEST, X_LSTG)
x = {k: v[d['idx_x'], :] for k, v in x_lstg.items() if k in sizes}
x[LSTG] = np.concatenate([x[LSTG], d['x'][THREAD]], axis=1)
for k, v in d['x'].items():
    if k != THREAD:
        x[k] = v

x = np.concatenate([x[k] for k in sizes.keys()], axis=1)

# turn number
dummies = d['x'][THREAD][:, -2:]
turn = 6 - 4 * dummies[:, 0] - 2 * dummies[:, 1]

# turn 2
x_t = x[turn == 2, :]
idx_rand = np.random.choice(range(np.shape(x_t)[0]), size=105, replace=False)
background = x_t[idx_rand[:100], :]
e = shap.KernelExplainer(lambda z: wrapper(model, z), background)
shap_vals = e.shap_values(x_t[idx_rand[100:], :])

# put in series
featnames = load_featnames(POLICY_SLR)
names = []
for k in sizes.keys():
    if not k.startswith('offer'):
        names += featnames[k]
    else:
        t = int(k[-1])
        offer_names = ['{}_{}'.format(n, t) for n in featnames['offer']]
        names += offer_names
for feat in TURN_FEATS[SLR]:
    names.remove(feat)
df = pd.DataFrame(shap_vals, columns=names)

# sum by feature group
cols = dict()
cols['w2v'] = ['{}{}'.format(name, num)
               for name in [BYR, SLR]
               for num in range(32)]
cols['cndtn'] = ['new', 'used', 'refurb', 'wear']
cols['slr'] = ['store', 'us', 'lstg_ct', 'bo_ct', 'fdbk_score', 'fdbk_pstv', 'fdbk_100']
cols['price'] = [c for c in featnames[LSTG]
                 if 'start_is' in c or 'start_price' in c or 'decline' in c]
cols['lstg'] = ['fast', 'photos', 'has_photos']
cols['clock'] = ['{}_{}'.format(feat, t)
                 for feat in CLOCK_FEATS
                 for t in range(1, 7)]
cols['clock'] += ['start_dow{}'.format(d) for d in range(6)]
cols['clock'] += ['start_years', 'start_holiday']
cols['byr'] = [BYR_HIST]
cols['tf'] = ['{}_{}'.format(feat, t) for feat in TIME_FEATS for t in range(1, 7)]
cols['tf'] += [THREAD_COUNT]
cols['delay'] = ['{}_{}'.format(feat, t) for feat in [DAYS, DELAY] for t in range(1, 7)]
cols['delay'] += [DAYS_SINCE_LSTG]
cols['offer'] = ['{}_{}'.format(feat, t) for feat in OUTCOME_FEATS for t in range(1, 7)
                 if feat not in [DAYS, DELAY]]
assert sum([len(c) for c in cols.values()]) == len(names)

groups = pd.DataFrame(index=df.index)
for k, v in cols.items():
    groups[k] = df[v].sum(axis=1)

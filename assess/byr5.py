import numpy as np
import pandas as pd
from statsmodels.nonparametric.kernel_regression import KernelReg
from agent.util import find_best_run, get_byr_agent, load_values
from utils import load_data, topickle
from assess.const import OPT
from constants import TEST, PLOT_DIR
from featnames import X_OFFER, CON, NORM, INDEX, LSTG, AUTO, EXP, BYR

MANUAL = 'manual'


# output dimensions
dim1 = np.arange(.65, .91, .01)
dim2 = np.arange(0.5, .86, .01)
xx1, xx2 = np.meshgrid(dim1, dim2)
mesh = np.concatenate([xx1.reshape(-1, 1), xx2.reshape(-1, 1)], axis=1)

# byr run
run_dir = find_best_run(byr=True, delta=.9)
data = load_data(part=TEST, run_dir=run_dir)
vals = load_values(part=TEST, delta=.9)

# restrict to agent threads
idx = get_byr_agent(data)
offers = pd.DataFrame(index=idx).join(data[X_OFFER])

byr5 = offers[CON].xs(5, level=INDEX)
slr = dict()
for t in [2, 4]:
    slr[t] = offers.xs(t, level=INDEX).reindex(
        index=byr5.index)[[AUTO, EXP, NORM]]
    slr[t].loc[:, NORM] = 1 - slr[t][NORM]
    slr[t]['manual'] = ~slr[t][AUTO] & ~slr[t][EXP]
    assert (slr[t][[MANUAL, AUTO, EXP]].sum(axis=1) == 1).all()

y = byr5.values.astype(np.float64)
x1 = slr[4][NORM].values
x2 = vals.reindex(index=slr[4].index, level=LSTG).values
x = np.stack([x1, x2], axis=1)

# univariate regressions for rejections
rej = slr[4][NORM] == 1
df = pd.DataFrame(columns=[MANUAL, AUTO, EXP],
                  index=pd.Index(dim2, name='value'))
for feat in [MANUAL, AUTO, EXP]:
    mask = rej & slr[2][feat] & slr[4][feat] & (x2 > .35)
    ll = KernelReg(y[mask], x2[mask], var_type='c', defaults=OPT)
    df[feat] = ll.fit(dim2)[0]
    print('{}: {}'.format(feat, ll.bw[0]))


# bivariate kernel regression
mask = (x1 > .55) & (x1 < 1) & (x2 > .4) & (x2 < .95)
ll2 = KernelReg(y[mask], x[mask, :], var_type='cc', defaults=OPT)
y_hat2 = ll2.fit(mesh)[0]
s2 = pd.Series(y_hat2, index=pd.MultiIndex.from_arrays(
    [mesh[:, 0], mesh[:, 1]], names=['norm', 'value']))

# put in dictionary
d = dict(normval=dict())
d['normval'] = s2

topickle(d, PLOT_DIR + '{}5.pkl'.format(BYR))

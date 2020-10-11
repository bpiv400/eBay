import numpy as np
import pandas as pd
from statsmodels.nonparametric.kernel_regression import KernelReg
from agent.util import find_best_run, get_byr_agent, load_values
from utils import load_data, topickle
from assess.const import OPT
from constants import TEST, PLOT_DIR
from featnames import X_OFFER, CON, NORM, INDEX, LSTG, DELAY, BYR

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

con = offers[CON].xs(3, level=INDEX)
last = offers.xs(2, level=INDEX).reindex(index=con.index)
norm = 1 - last[NORM]
norm[(last[CON] == 0) & (last[DELAY] == 0)] = 1.05
norm[(last[CON] == 0) & (last[DELAY] == 1)] = 1.1
y = con.values.astype(np.float64)
x1 = norm.values
x2 = vals.reindex(index=norm.index, level=LSTG).values
x = np.stack([x1, x2], axis=1)

# univariate regressions for rejections
cols = {'manual': 1, 'auto': 1.05, 'exp': 1.1}
df = pd.DataFrame(columns=cols, index=pd.Index(dim2, name='value'))
for k, v in cols.items():
    mask = np.isclose(x1, v) & (x2 > .42) & (x2 < .93)
    ll = KernelReg(y[mask], x2[mask], var_type='c', defaults=OPT)
    df[k] = ll.fit(dim2)[0]
    print('{}: {}'.format(k, ll.bw[0]))

# bivariate kernel regression
mask = (x1 > .55) & (x1 < 1) & (x2 > .4) & (x2 < .95)
ll2 = KernelReg(y[mask], x[mask, :], var_type='cc', defaults=OPT)
y_hat2 = ll2.fit(mesh)[0]
s2 = pd.Series(y_hat2, index=pd.MultiIndex.from_arrays(
    [mesh[:, 0], mesh[:, 1]], names=['norm', 'value']))

# put in dictionary
d = dict(normval=dict())
d['normval'] = s2

topickle(d, PLOT_DIR + '{}3.pkl'.format(BYR))

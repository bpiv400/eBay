import numpy as np
import pandas as pd
from statsmodels.nonparametric.kernel_regression import KernelReg
from agent.util import find_best_run, get_byr_agent, load_values
from utils import load_data, topickle, safe_reindex
from assess.const import OPT
from constants import TEST, PLOT_DIR
from featnames import X_OFFER, CON, NORM, INDEX, LSTG, AUTO, EXP, BYR

data = load_data(part=TEST)

# restrict to 50% opening offers
mask = data[X_OFFER][CON].xs(1, level=INDEX) == .5
idx = mask[mask].index
for k, v in data.items():
    data[k] = safe_reindex(v, idx=idx)



# output dimensions
dim1 = np.arange(.65, .91, .01)
dim2 = np.arange(0.5, .86, .01)
xx1, xx2 = np.meshgrid(dim1, dim2)
mesh = np.concatenate([xx1.reshape(-1, 1), xx2.reshape(-1, 1)], axis=1)

# byr run

vals = load_values(part=TEST, delta=.9)

slr4 = offers[CON].xs(4, level=INDEX)
byr = dict()
for t in [1, 3]:
    byr[t] = offers.xs(t, level=INDEX).reindex(
        index=slr4.index)[[AUTO, EXP, NORM]]
    slr[t].loc[:, NORM] = 1 - slr[t][NORM]
    slr[t]['manual'] = ~slr[t][AUTO] & ~slr[t][EXP]
    assert (slr[t][[MANUAL, AUTO, EXP]].sum(axis=1) == 1).all()


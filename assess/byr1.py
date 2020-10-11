import numpy as np
import pandas as pd
from processing.util import hist_to_pctile
from assess.util import ll_wrapper
from utils import load_data, topickle, safe_reindex
from assess.const import DISCRETE
from constants import TEST, PLOT_DIR
from featnames import X_OFFER, X_THREAD, BYR_HIST, CON, INDEX, BYR

d = dict()

# distributions of byr_hist for those who make 50% concessions
data = load_data(part=TEST)

con1 = data[X_OFFER].xs(1, level=INDEX)[CON]
hist1 = data[X_THREAD][BYR_HIST]
assert np.all(con1.index == hist1.index)
y, x = hist1.values, con1.values

dim = np.linspace(.4, 1, 61)  # opening offers
mask = x > .33
d['response_hist'] = ll_wrapper(y[mask], x[mask], dim=dim, discrete=DISCRETE)

dim = np.linspace(0, 1, 101)
for t in [3, 5]:
    con = data[X_OFFER].xs(t, level=INDEX)[CON]
    hist = hist1.reindex(index=con.index)
    d['response_hist{}'.format(t)] = \
        ll_wrapper(hist.values, con.values, dim=dim, discrete=DISCRETE)
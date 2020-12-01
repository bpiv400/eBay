import pandas as pd
from agent.util import find_best_run, load_valid_data, load_values, get_byr_agent
from assess.util import continuous_cdf
from utils import safe_reindex, topickle
from constants import PLOT_DIR
from featnames import LOOKUP, X_OFFER, INDEX, THREAD, CON, NORM, TEST

DELTA_FIGURES = .99


d = {}

vals = load_values(part=TEST, delta=DELTA_FIGURES)
run_dir = find_best_run(byr=True, delta=DELTA_FIGURES)
data = load_valid_data(part=TEST, run_dir=run_dir)
valid_vals = safe_reindex(vals, idx=data[LOOKUP].index)

threads = get_byr_agent(data)
df = safe_reindex(data[X_OFFER], idx=threads)

rej7 = df[CON].xs(7, level=INDEX) == 0
norm = (1 - df[NORM].xs(6, level=INDEX)).reindex(index=rej7.index)
rej7 = rej7.droplevel(THREAD)
norm = norm.droplevel(THREAD)
other = rej7[~rej7].index
reject = rej7[rej7].index
reward = valid_vals.reindex(index=norm.index) - norm
elem = {'Reject': continuous_cdf(reward.loc[reject]),
        'Accept': continuous_cdf(reward.loc[other])}
d['cdf_netvalue'] = pd.DataFrame.from_dict(elem)

topickle(d, PLOT_DIR + 'byr7.pkl')

import numpy as np
import pandas as pd
from statsmodels.nonparametric.kde import KDEUnivariate
from agent.util import find_best_run, load_valid_data, load_values, get_byr_agent
from assess.util import continuous_cdf
from utils import topickle, get_role, safe_reindex
from agent.const import DELTA_CHOICES
from constants import PLOT_DIR, TEST, EPS
from featnames import LOOKUP, X_OFFER, INDEX, THREAD, CON, NORM

DIM = np.linspace(1/1000, 1, 1000)


def transform(x):
    z = np.clip(x, EPS, 1 - EPS)
    return np.log(z / (1 - z))


def kdens(x, dim=DIM):
    f = KDEUnivariate(transform(x))
    f.fit(kernel='gau', bw='silverman', fft=True)
    f_hat = f.evaluate(transform(dim))
    return f_hat


def main():
    d, df_delta, prefix = {}, pd.DataFrame(index=DIM), 'cdf_values'
    for delta in DELTA_CHOICES:
        vals = load_values(part=TEST, delta=delta)
        df_delta['$\\delta = {}$'.format(delta)] = kdens(vals)

        for byr in [True, False]:
            key = '{}_{}_{}'.format(prefix, get_role(byr), delta)

            run_dir = find_best_run(byr=byr, delta=delta)
            if run_dir is None:
                continue
            data = load_valid_data(part=TEST, run_dir=run_dir)
            valid_vals = safe_reindex(vals, idx=data[LOOKUP].index)

            # distributions of values
            elem = {'All': continuous_cdf(vals),
                    'Valid': continuous_cdf(valid_vals)}
            d[key] = pd.DataFrame.from_dict(elem)

            if byr:
                threads = get_byr_agent(data)
                df = safe_reindex(data[X_OFFER], idx=threads)

                # turn 1
                rej1 = (df[CON].xs(1, level=INDEX) == 0).droplevel(THREAD)
                other = rej1[~rej1].index
                reject = rej1[rej1].index
                elem = {'Reject': continuous_cdf(valid_vals.loc[reject]),
                        'Concession': continuous_cdf(valid_vals.loc[other])}
                d['{}_turn1'.format(key)] = pd.DataFrame.from_dict(elem)

                # turn 7
                rej7 = df[CON].xs(7, level=INDEX) == 0
                norm = (1 - df[NORM].xs(6, level=INDEX)).reindex(index=rej7.index)
                rej7 = rej7.droplevel(THREAD)
                norm = norm.droplevel(THREAD)
                other = rej7[~rej7].index
                reject = rej7[rej7].index
                reward = valid_vals.reindex(index=norm.index) - norm
                elem = {'Reject': continuous_cdf(reward.loc[reject]),
                        'Accept': continuous_cdf(reward.loc[other])}
                d['cdf_netvalue_turn7'] = pd.DataFrame.from_dict(elem)

    d['pdf_values'] = df_delta

    # save
    topickle(d, PLOT_DIR + 'values.pkl')


if __name__ == '__main__':
    main()

import pandas as pd
from agent.util import get_byr_agent, load_values, find_best_run, load_valid_data
from assess.util import continuous_cdf, kdens_wrapper
from utils import safe_reindex, topickle
from agent.const import DELTA_BYR
from constants import PLOT_DIR
from featnames import X_OFFER, CON, INDEX, THREAD, TEST, LOOKUP, NORM


def compare_rewards(walk=None, other=None, y=None, action='Offer'):
    if len(walk) == 0:
        elem = {action: continuous_cdf(y.loc[other])}
    elif len(other) == 0:
        elem = {'Walk': continuous_cdf(y.loc[walk])}
    else:
        elem = {'Walk': continuous_cdf(y.loc[walk]),
                action: continuous_cdf(y.loc[other])}
    return pd.DataFrame.from_dict(elem)


def main():
    d = dict()

    # values when buyer arrives
    for delta in DELTA_BYR:
        vals = load_values(part=TEST, delta=delta)
        run_dir = find_best_run(byr=True, delta=delta)
        if run_dir is None:
            continue
        data = load_valid_data(part=TEST, run_dir=run_dir)
        valid_vals = safe_reindex(vals, idx=data[LOOKUP].index)

        # comparing all to valid values
        kwargs = {'All': vals, 'Valid': valid_vals}
        d['pdf_values_{}'.format(delta)] = kdens_wrapper(**kwargs)

        # agent offers
        threads = get_byr_agent(data)
        df = safe_reindex(data[X_OFFER][[CON, NORM]], idx=threads)

        # first-turn decision
        rej1 = (df[CON].xs(1, level=INDEX) == 0).droplevel(THREAD)
        d['cdf_t1value_{}'.format(delta)] = compare_rewards(
            walk=rej1[rej1].index, other=rej1[~rej1].index, y=valid_vals)

        # last-turn decision
        rej7 = df[CON].xs(7, level=INDEX) == 0
        norm6 = (1 - df[NORM].xs(6, level=INDEX)).reindex(index=rej7.index)
        rej7 = rej7.droplevel(THREAD)
        norm6 = norm6.droplevel(THREAD)
        reward = valid_vals.reindex(index=norm6.index) - norm6
        d['cdf_t7value_{}'.format(delta)] = compare_rewards(
            walk=rej7[rej7].index, other=rej7[~rej7].index, y=reward, action='Accept')

    # save
    topickle(d, PLOT_DIR + 'byrvals.pkl')


if __name__ == '__main__':
    main()

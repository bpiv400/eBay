import pandas as pd
from agent.util import get_byr_agent, load_values, find_best_run, load_valid_data
from assess.util import continuous_cdf, kdens_wrapper
from utils import safe_reindex, topickle
from agent.const import DELTA_BYR
from constants import PLOT_DIR
from featnames import X_OFFER, CON, INDEX, THREAD, TEST, LOOKUP, NORM


def compare_rewards(walk=None, offer=None, accept=None, y=None):
    elem = dict()
    if walk is not None and len(walk) > 0:
        elem['Walk'] = continuous_cdf(y.loc[walk])
    if offer is not None and len(offer) > 0:
        elem['Offer'] = continuous_cdf(y.loc[offer])
    if accept is not None and len(accept) > 0:
        elem['Accept'] = continuous_cdf(y.loc[accept])
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
        df = safe_reindex(data[X_OFFER][[CON, NORM]],
                          idx=get_byr_agent(data)).droplevel(THREAD)

        # first-turn decision
        con1 = df[CON].xs(1, level=INDEX)
        d['cdf_t1value_{}'.format(delta)] = compare_rewards(
            walk=con1[con1 == 0].index,
            offer=con1[(con1 > 0) & (con1 < 1)].index,
            accept=con1[con1 == 1].index,
            y=valid_vals
        )

        # last-turn decision
        con7 = df[CON].xs(7, level=INDEX) == 0
        norm6 = (1 - df[NORM].xs(6, level=INDEX)).reindex(index=con7.index)
        reward = valid_vals.reindex(index=norm6.index) - norm6
        d['cdf_t7value_{}'.format(delta)] = compare_rewards(
            walk=con7[con7 == 0].index,
            accept=con7[con7 == 1].index,
            y=reward
        )

    # save
    topickle(d, PLOT_DIR + 'byrvals.pkl')


if __name__ == '__main__':
    main()

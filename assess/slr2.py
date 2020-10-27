import numpy as np
import pandas as pd
from agent.util import load_values, find_best_run, load_valid_data, \
    get_norm_reward, get_slr_valid, get_sale_norm
from assess.util import kdens_wrapper, kreg2, ll_wrapper, continuous_cdf
from utils import topickle, safe_reindex, load_data
from agent.const import DELTA_CHOICES
from assess.const import NORM1_DIM, SPLITS
from constants import PLOT_DIR
from featnames import X_OFFER, CON, INDEX, TEST, LOOKUP, AUTO, EXP, REJECT, \
    SLR_BO_CT


def main():
    d = dict()

    obs = get_slr_valid(load_data(part=TEST))

    # value comparison
    for delta in DELTA_CHOICES:
        run_dir = find_best_run(byr=False, delta=delta)
        if run_dir is None:
            continue
        data = load_valid_data(part=TEST, run_dir=run_dir)
        vals = load_values(part=TEST, delta=delta)

        valid_vals = safe_reindex(vals, idx=data[LOOKUP].index)
        kwargs = {'All': vals, 'Valid': valid_vals}
        d['pdf_values_{}'.format(delta)] = kdens_wrapper(**kwargs)

        sale_norm_obs, cont_value_obs = \
            get_norm_reward(data=obs, values=vals * delta)
        sale_norm_agent, cont_value_agent = \
            get_norm_reward(data=data, values=vals * delta)

        # cdf of values
        elem = {'Data': continuous_cdf(sale_norm_obs.append(cont_value_obs)),
                'Agent': continuous_cdf(sale_norm_agent.append(cont_value_agent))}
        d['cdf_realval_{}'.format(delta)] = pd.DataFrame(elem)

        # cdf of continuation values
        elem = {'Data': continuous_cdf(cont_value_obs / delta),
                'Agent': continuous_cdf(cont_value_agent / delta)}
        d['cdf_contval_{}'.format(delta)] = pd.DataFrame(elem)

    # # seller experience
    # obs = load_data(part=TEST)
    # offers2 = obs[X_OFFER][[CON, EXP, AUTO, REJECT]].xs(2, level=INDEX)
    # offers2['active'] = offers2[REJECT] & ~offers2[AUTO] & ~offers2[EXP]
    # con1 = obs[X_OFFER][CON].xs(1, level=INDEX).loc[offers2.index]
    # hist = safe_reindex(obs[LOOKUP][SLR_BO_CT], idx=offers2.index)
    #
    # x_con1, x_hist = con1.values, hist.values
    # for name in ['active', EXP, AUTO]:
    #     print(name)
    #     d['contour_hist{}'.format(name)] = \
    #         kreg2(y=offers2[name].values, x1=x_con1, x2=x_hist)

    topickle(d, PLOT_DIR + 'slr2.pkl')


if __name__ == '__main__':
    main()

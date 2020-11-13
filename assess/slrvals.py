import pandas as pd
from agent.util import load_values, find_best_run, load_valid_data, \
    get_norm_reward, get_slr_valid
from assess.util import kdens_wrapper, continuous_cdf
from utils import topickle, safe_reindex, load_data
from agent.const import DELTA_CHOICES
from constants import PLOT_DIR
from featnames import TEST, LOOKUP


def main():
    d = dict()

    obs = get_slr_valid(load_data(part=TEST))

    # value comparison
    for delta in DELTA_CHOICES:
        run_dir = find_best_run(byr=False, delta=delta)
        data = load_valid_data(part=TEST, run_dir=run_dir)
        vals = load_values(part=TEST, delta=delta)

        valid_vals = safe_reindex(vals, idx=data[LOOKUP].index)
        kwargs = {'All': vals, 'Valid': valid_vals}
        d['pdf_values_{}'.format(delta)] = kdens_wrapper(**kwargs)

        sale_norm_obs, cont_value_obs = \
            get_norm_reward(data=obs, values=(vals * delta))
        sale_norm_agent, cont_value_agent = \
            get_norm_reward(data=data, values=(vals * delta))

        # cdf of values
        elem = {'Data': continuous_cdf(sale_norm_obs.append(cont_value_obs)),
                'Agent': continuous_cdf(sale_norm_agent.append(cont_value_agent))}
        d['cdf_realval_{}'.format(delta)] = pd.DataFrame(elem)

        # cdf of values for unsold items
        elem = {'Data': cont_value_obs / delta,
                'Agent': cont_value_agent / delta}
        d['pdf_unsoldvals_{}'.format(delta)] = kdens_wrapper(**elem)

        # cdf of values for unsold items
        elem = {'Data': safe_reindex(vals, idx=sale_norm_obs.index),
                'Agent': safe_reindex(vals, idx=sale_norm_agent.index)}
        d['pdf_soldvals_{}'.format(delta)] = kdens_wrapper(**elem)

    # save
    topickle(d, PLOT_DIR + 'slrvals.pkl')


if __name__ == '__main__':
    main()

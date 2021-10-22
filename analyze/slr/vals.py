from agent.util import load_values, get_sim_dir, load_valid_data, get_norm_reward
from analyze.util import create_cdfs, save_dict
from utils import safe_reindex
from agent.const import DELTA_SLR
from featnames import LOOKUP


def main():
    d = dict()

    obs = load_valid_data(byr=False)

    # value comparison
    for delta in DELTA_SLR:
        sim_dir = get_sim_dir(delta=delta)
        data = load_valid_data(sim_dir=sim_dir)
        vals = load_values(delta=delta)

        valid_vals = safe_reindex(vals, idx=data[LOOKUP].index)
        elem = {'All': vals, 'Valid': valid_vals}
        d['cdf_values_{}'.format(delta)] = create_cdfs(elem)

        sale_norm_obs, cont_value_obs = \
            get_norm_reward(data=obs, values=vals)
        sale_norm_agent, cont_value_agent = \
            get_norm_reward(data=data, values=vals)

        # cdf of values
        elem = {'Data': sale_norm_obs.append(cont_value_obs * delta),
                'Agent': sale_norm_agent.append(cont_value_agent * delta)}
        d['cdf_realval_{}'.format(delta)] = create_cdfs(elem)

        # cdf of values for unsold items
        elem = {'Data': cont_value_obs, 'Agent': cont_value_agent}
        d['cdf_unsoldvals_{}'.format(delta)] = create_cdfs(elem)

        # cdf of values for unsold items
        elem = {'Data': safe_reindex(vals, idx=sale_norm_obs.index),
                'Agent': safe_reindex(vals, idx=sale_norm_agent.index)}
        d['cdf_soldvals_{}'.format(delta)] = create_cdfs(elem)

    # save
    save_dict(d, 'slrvals')


if __name__ == '__main__':
    main()

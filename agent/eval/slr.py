import pandas as pd
from agent.util import load_values, get_norm_reward, get_sim_dir, load_valid_data
from agent.eval.util import eval_args, get_eval_path
from utils import unpickle, topickle
from agent.const import DELTA_SLR, SLR_NAMES
from featnames import LOOKUP, START_PRICE


def get_return(data=None, values=None):
    sale_norm, cont_value = get_norm_reward(data=data, values=values)
    norm = pd.concat([sale_norm, cont_value]).sort_index()
    start_price = data[LOOKUP][START_PRICE]
    dollar = norm * start_price
    net_norm = norm - values
    dollar_norm = net_norm * start_price
    s = pd.Series()
    s['norm'] = norm.mean()
    s['dollar'] = dollar.mean()
    s['norm_sale'] = sale_norm.mean()
    s['dollar_sale'] = dollar.loc[sale_norm.index].mean()
    s['sale_pct'] = len(sale_norm) / (len(sale_norm) + len(cont_value))
    s['net_norm'] = net_norm.mean()
    s['dollar_norm'] = dollar_norm.mean()
    return s


def get_output(d=None, delta=None):
    k = SLR_NAMES[delta]
    print(k)
    if k not in d:
        d[k] = pd.DataFrame()

    values = 0 if delta == 0 else delta * load_values(delta=delta)

    # humans
    data_obs = load_valid_data(byr=False, minimal=True)
    d[k]['Humans'] = get_return(data=data_obs, values=values)

    # heuristic
    h_dir = get_sim_dir(byr=False, delta=delta, heuristic=True)
    data_h = load_valid_data(sim_dir=h_dir, minimal=True)
    d[k]['Heuristic'] = get_return(data=data_h, values=values)

    # agent
    sim_dir = get_sim_dir(byr=False, delta=delta)
    data_rl = load_valid_data(sim_dir=sim_dir, minimal=True)
    d[k]['Agent'] = get_return(data=data_rl, values=values)


def main():
    args = eval_args()

    path = get_eval_path(byr=False)
    if args.read:
        d = unpickle(path)
        for k, v in d.items():
            print(k)
            print(v)
        exit()

    # create output statistics
    d = dict()
    for delta in DELTA_SLR:
        get_output(d, delta=delta)

    # save
    topickle(d, path)


if __name__ == '__main__':
    main()

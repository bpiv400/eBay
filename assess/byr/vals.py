from agent.util import get_byr_agent, load_values, get_run_dir, \
    load_valid_data, get_sale_norm
from assess.util import create_cdfs
from utils import safe_reindex, topickle
from agent.const import DELTA_BYR, COMMON_CONS
from constants import PLOT_DIR, IDX
from featnames import X_OFFER, CON, INDEX, THREAD, TEST, LOOKUP, NORM, BYR, SIM


def compare_rewards(walk=None, offer=None, accept=None, y=None):
    elem = dict()
    if walk is not None and len(walk) > 0:
        elem['Walk'] = y.loc[walk]
    if offer is not None and len(offer) > 0:
        elem['Offer'] = y.loc[offer]
    if accept is not None and len(accept) > 0:
        elem['Accept'] = y.loc[accept]
    return create_cdfs(elem)


def compare_by_turn(df=None, turn=None, values=None):
    con = df[CON].xs(turn, level=INDEX)
    if turn == 1:
        reward = values
    elif turn in [3, 5]:
        smallest = COMMON_CONS[1][0] + (1 - COMMON_CONS[1][0]) * COMMON_CONS[3][0]
        if turn == 5:
            smallest += (1 - smallest) * COMMON_CONS[5][0]
        reward = values.reindex(index=con.index) - smallest
    else:
        norm = (1 - df[NORM].xs(turn - 1, level=INDEX)).reindex(index=con.index)
        reward = values.reindex(index=con.index) - norm
    cdfs = compare_rewards(
        walk=con[con == 0].index,
        offer=con[(con > 0) & (con < 1)].index,
        accept=con[con == 1].index,
        y=reward
    )
    return cdfs


def split_by_bought(data=None):
    bought = get_sale_norm(data[X_OFFER], drop_thread=False).xs(
        1, level=THREAD).index
    unbought = data[LOOKUP].index.drop(bought)
    if SIM in bought.names:
        bought = bought.droplevel(SIM)
        unbought = unbought.droplevel(SIM)
    return bought, unbought


def main():
    d = dict()

    data_obs = load_valid_data(part=TEST, byr=True)
    bought_obs, unbought_obs = split_by_bought(data=data_obs)

    # values when buyer arrives
    for delta in DELTA_BYR:
        vals = load_values(part=TEST, delta=delta)
        run_dir = get_run_dir(byr=True, delta=delta)
        if run_dir is None:
            continue
        data = load_valid_data(part=TEST, run_dir=run_dir, byr=True)
        valid_vals = safe_reindex(vals, idx=data[LOOKUP].index)

        # comparing all to valid values
        elem = {'All': vals, 'Valid': valid_vals}
        d['cdf_values_{}'.format(delta)] = create_cdfs(elem)

        # comparing values of bought and unbought items
        bought, unbought = split_by_bought(data=data)
        elem = {'Data': vals.loc[bought_obs], 'Agent': vals.loc[bought]}
        d['cdf_soldvals_{}'.format(delta)] = create_cdfs(elem)
        elem = {'Data': vals.loc[unbought_obs], 'Agent': vals.loc[unbought]}
        d['cdf_unsoldvals_{}'.format(delta)] = create_cdfs(elem)

        # agent offers
        df = safe_reindex(data[X_OFFER][[CON, NORM]],
                          idx=get_byr_agent(data)).droplevel(THREAD)

        # first-turn decision
        for turn in IDX[BYR]:
            d['cdf_t{}value_{}'.format(turn, delta)] = \
                compare_by_turn(df=df, turn=turn, values=valid_vals)

    # save
    topickle(d, PLOT_DIR + 'byrvals.pkl')


if __name__ == '__main__':
    main()

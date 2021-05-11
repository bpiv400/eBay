from agent.util import load_values, load_valid_data, get_run_dir, get_sale_norm
from utils import safe_reindex, load_data
from agent.const import DELTA_SLR
from assess.const import SLR_NAMES
from constants import IDX
from featnames import LOOKUP, X_THREAD, X_OFFER, LSTG, INDEX, NORM, REJECT, CON, SLR


def main():
    valid_data = load_valid_data(byr=False, minimal=True)
    data = load_data()

    # valid vs. invalid listings
    lstgs = data[LOOKUP].index
    valid = valid_data[LOOKUP].index
    invalid = lstgs.drop(valid)

    values = load_values(delta=DELTA_SLR[-1])
    valid_vals = safe_reindex(values, idx=valid)

    print('Average value (all listings): {}'.format(values.mean()))
    print('Average value (valid listings): {}'.format(valid_vals.mean()))
    print('Share of listings that are valid: {}'.format(
        len(valid_vals) / len(values)))

    num_threads = data[X_THREAD].iloc[:, 0].groupby(LSTG).count().reindex(
        index=lstgs, fill_value=0)
    print('Share of invalid listings without an arrival: {}'.format(
        (num_threads.loc[invalid] == 0).mean()))

    # sales
    sale_norm = get_sale_norm(offers=valid_data[X_OFFER]).reindex(
        index=valid, fill_value=0)
    offers = valid_data[X_OFFER]
    norm1 = offers[NORM].xs(1, level=INDEX)
    norm3 = offers.loc[~offers[REJECT], NORM].xs(3, level=INDEX)
    print('Share of valid listings in data that do not sell: {}'.format(
        (sale_norm == 0).mean()))
    print('Share of valid listings in data that sell for list price: {}'.format(
        (sale_norm == 1).mean()))
    for t in [1, 3]:
        print('Share of turn {} for list price in valid data: {}'.format(
            t, (locals()['{}{}'.format(NORM, t)] == 1).mean()))

    # small concessions by sellers
    con = data[X_OFFER].loc[data[X_OFFER].index.isin(IDX[SLR], level=INDEX), CON]
    con = con[(con > 0) & (con < 1)]
    print('con < 0.5): {}'.format((con < .5).mean()))

    # for the agent sellers
    for delta in DELTA_SLR:
        run_dir = get_run_dir(delta=delta)
        data_rl = load_valid_data(sim_dir=run_dir)
        sale_norm_rl = get_sale_norm(offers=data_rl[X_OFFER]).reindex(
            index=data_rl[LOOKUP].index, fill_value=0)
        offers_rl = data_rl[X_OFFER]
        norm1_rl = offers_rl[NORM].xs(1, level=INDEX)
        norm3_rl = offers_rl.loc[~offers_rl[REJECT], NORM].xs(3, level=INDEX)
        print('{}. Share of listings that do not sell: {}'.format(
            SLR_NAMES[delta], (sale_norm_rl == 0).mean()))
        print('{}. Share of listings that sell for list price: {}'.format(
            SLR_NAMES[delta], (sale_norm_rl == 1).mean()))
        for t in [1, 3]:
            print('{}. Share of turn {} for list price in valid data: {}'.format(
                SLR_NAMES[delta], t, (locals()['{}{}_rl'.format(NORM, t)] == 1).mean()))


if __name__ == '__main__':
    main()

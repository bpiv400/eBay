from agent.util import load_values, load_valid_data, get_sim_dir, get_sale_norm
from utils import safe_reindex, load_data, load_feats
from agent.const import DELTA_SLR
from assess.const import SLR_NAMES
from constants import IDX
from featnames import LOOKUP, X_THREAD, X_OFFER, LSTG, INDEX, NORM, REJECT, CON, SLR, \
    STORE, SLR_BO_CT


def print_sales_stats(data=None):
    sale_norm = get_sale_norm(offers=data[X_OFFER]).reindex(
        index=data[LOOKUP].index, fill_value=0)
    offers = data[X_OFFER]
    norm1 = offers[NORM].xs(1, level=INDEX)
    norm3 = offers.loc[~offers[REJECT], NORM].xs(3, level=INDEX)
    print('Share of valid listings in data that do not sell: {}'.format(
        (sale_norm == 0).mean()))
    print('Share of valid listings in data that sell for list price: {}'.format(
        (sale_norm == 1).mean()))
    for t in [1, 3]:
        print('Share of turn {} for list price in valid data: {}'.format(
            t, (locals()['{}{}'.format(NORM, t)] == 1).mean()))


def main():
    data = load_data()
    lstgs = data[LOOKUP].index
    lstg_feats = load_feats('listings', lstgs=lstgs)[[SLR, STORE, SLR_BO_CT]]

    # correlation between store and experience
    df = lstg_feats.groupby(SLR).max()
    print('Median number of BOs for sellers w/store: {}'.format(
        df.loc[df[STORE], SLR_BO_CT].median()))
    print('Median number of BOs for sellers w/o store: {}'.format(
        df.loc[~df[STORE], SLR_BO_CT].median()))

    # valid vs. invalid listings
    valid_data = load_valid_data(byr=False, minimal=True)
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
    print_sales_stats(valid_data)

    # small concessions by sellers
    con = data[X_OFFER].loc[data[X_OFFER].index.isin(IDX[SLR], level=INDEX), CON]
    con = con[(con > 0) & (con < 1)]
    print('con < .5: {}'.format((con < .5).mean()))

    store = safe_reindex(lstg_feats[STORE], idx=con.index)
    print('Store, con < .5: {}'.format((con[store] < .5).mean()))
    print('No store, con < .5: {}'.format((con[~store] < .5).mean()))

    # for the agent sellers
    for delta in DELTA_SLR:
        print('{}:'.format(SLR_NAMES[delta]))
        sim_dir = get_sim_dir(byr=False, delta=delta)
        data_rl = load_valid_data(sim_dir=sim_dir)
        print_sales_stats(data_rl)


if __name__ == '__main__':
    main()

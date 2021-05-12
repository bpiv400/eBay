from agent.util import load_valid_data, get_sim_dir, get_sale_norm
from utils import safe_reindex
from featnames import X_OFFER, CON, AUTO, INDEX, LOOKUP, START_PRICE, THREAD


def share_accept(data=None):
    con = data[X_OFFER].xs(1, level=INDEX)[CON]
    print('Share of non-BIN first offers for half of list price: {}'.format(
        (con == .5).sum() / len(con[con < 1])))

    idx = con[con == .5].index
    con2 = safe_reindex(data[X_OFFER][CON].xs(2, level=INDEX), idx=idx)
    con2.loc[con2.isna()] = 0  # when seller does not respond
    print('Seller accept rate: {}'.format((con2 == 1).mean()))

    idx_acc = con2[con2 == 1].index
    auto2 = safe_reindex(data[X_OFFER][AUTO].xs(2, level=INDEX), idx=idx)
    print('Share of seller accepts that are automatic: {}'.format(
        auto2.loc[idx_acc].mean()))

    idx_acc = idx_acc.droplevel(THREAD)
    start_price = data[LOOKUP].loc[idx_acc, START_PRICE]
    num = start_price / 2

    sale_norm = get_sale_norm(offers=data[X_OFFER], drop_thread=False)
    sale_norm = safe_reindex(sale_norm, idx=idx).dropna()
    sale_norm.index = sale_norm.index.droplevel(THREAD)
    den = (1 - sale_norm) * safe_reindex(data[LOOKUP][START_PRICE],
                                         idx=sale_norm.index)

    print('Reward share: {}'.format(num.sum() / den.sum()))


def main():
    data_obs = load_valid_data(byr=True, minimal=True)
    share_accept(data=data_obs)

    sim_dir = get_sim_dir(byr=True, delta=1)
    data_rl = load_valid_data(sim_dir=sim_dir, minimal=True)
    share_accept(data=data_rl)


if __name__ == '__main__':
    main()

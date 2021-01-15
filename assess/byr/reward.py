from agent.util import only_byr_agent, load_valid_data, get_run_dir, \
    load_values, get_sale_norm
from utils import safe_reindex
from agent.const import DELTA_BYR
from featnames import X_OFFER, CON, AUTO, INDEX, TEST, LOOKUP, START_PRICE, THREAD


def share_accept(data=None, vals=None):
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

    if vals is not None:
        idx_acc = idx_acc.droplevel(THREAD)
        num = safe_reindex(vals, idx=idx_acc) \
            - .5 * data[LOOKUP].loc[idx_acc, START_PRICE]

        sale_norm = get_sale_norm(offers=data[X_OFFER], drop_thread=False)
        sale_norm = safe_reindex(sale_norm, idx=idx).dropna()
        sale_norm.index = sale_norm.index.droplevel(THREAD)
        sale_price = sale_norm * safe_reindex(data[LOOKUP][START_PRICE],
                                              idx=sale_norm.index)
        den = safe_reindex(vals, idx=sale_price.index) - sale_price

        print('Reward share: {}'.format(num.sum() / den.sum()))


def main():
    vals = load_values(part=TEST, delta=DELTA_BYR[-1], normalize=False)

    data = only_byr_agent(load_valid_data(part=TEST, byr=True))
    share_accept(data=data, vals=vals)

    run_dir = get_run_dir(byr=True, delta=DELTA_BYR[-1])
    data = only_byr_agent(load_valid_data(part=TEST, byr=True, run_dir=run_dir))

    share_accept(data=data, vals=vals)


if __name__ == '__main__':
    main()

import pandas as pd
from agent.util import load_values
from assess.util import get_sale_norm
from utils import load_data, load_feats
from agent.const import DELTA_BYR
from featnames import TEST, X_OFFER, LOOKUP, START_PRICE, DEC_PRICE, STORE


def main():
    data = load_data(part=TEST)
    sale_norm = get_sale_norm(data[X_OFFER])
    psale = pd.Series(True, index=sale_norm.index).reindex(
        index=data[LOOKUP].index, fill_value=False)

    # decline price
    high = (data[LOOKUP][DEC_PRICE] / data[LOOKUP][START_PRICE]) > .5
    vals = load_values(part=TEST, delta=DELTA_BYR[-1])
    print('Avg sale price for decline price > .5: {}'.format(
        sale_norm.reindex(high[high].index).mean()))
    print('Avg sale price for decline price <= .5: {}'.format(
        sale_norm.reindex(high[~high].index).mean()))
    print('Avg value for decline price > .5: {}'.format(vals[high].mean()))
    print('Avg value for decline price <= .5: {}'.format(vals[~high].mean()))

    # store
    store = load_feats('listings').loc[data[LOOKUP].index][STORE]
    vals = load_values(part=TEST, delta=DELTA_BYR[0])
    print('Pr(sale) for store: {}'.format(psale[store].mean()))
    print('Pr(sale) for ~store: {}'.format(psale[~store].mean()))
    print('Avg sale price for store: {}'.format(
        sale_norm.reindex(store[store].index).mean()))
    print('Avg sale price for ~store: {}'.format(
        sale_norm.reindex(store[~store].index).mean()))
    print('Avg value for store: {}'.format(vals[store].mean()))
    print('Avg value for ~store: {}'.format(vals[~store].mean()))


if __name__ == '__main__':
    main()

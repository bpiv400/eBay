import numpy as np
import pandas as pd
from agent.util import load_values
from assess.util import get_sale_norm, kdens_wrapper, ll_wrapper, \
    calculate_row, get_dim
from processing.util import do_rounding, extract_day_feats
from utils import unpickle, topickle, load_data, load_feats
from agent.const import DELTA_CHOICES
from assess.const import DELTA_SHAP, LOG10_BIN_DIM
from constants import PLOT_DIR, PCTILE_DIR, DAY
from featnames import LOOKUP, X_OFFER, FDBK_SCORE, STORE, TEST, START_PRICE


def bin_plot(start_price=None, vals=None):
    x = start_price.values
    is_round, _ = do_rounding(x)
    x = np.log10(x)
    y = vals.values
    discrete = np.unique(x[is_round])
    line, dots, _ = ll_wrapper(y=y, x=x,
                               dim=LOG10_BIN_DIM,
                               discrete=discrete,
                               bw=(.05,))
    return line, dots


def dow_plot(listings=None, vals=None):
    dow = extract_day_feats(listings.start_date * DAY).drop('holiday', axis=1)
    dow['dow6'] = dow.sum(axis=1) == 0
    dow.columns = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                   'Friday', 'Saturday', 'Sunday']
    df = pd.DataFrame(index=dow.columns, columns=['beta', 'err'])
    for day in dow.columns:
        df.loc[day, :] = calculate_row(vals[dow[day]])
    return df


def photos_plot(listings=None, vals=None):
    photos = pctile_to_count(listings=listings, featname='photos')
    dim = range(int(photos.max() + 1))
    df = pd.DataFrame(index=dim, columns=['beta', 'err'])
    for i in dim:
        df.loc[i, :] = calculate_row(vals[photos == i])
    return df


def run_ll(y=None, x=None):
    dim = get_dim(x)
    line, _ = ll_wrapper(y=y, x=x, dim=dim, bw=(.05,))
    return line


def two_ll(y=None, x=None, mask=None, keys=None):
    drop = np.isclose(y, -99)
    x, y, mask = x[~drop], y[~drop], mask[~drop]
    d = dict()
    d[keys[0]] = run_ll(y=y[mask], x=x[mask])
    d[keys[1]] = run_ll(y=y[~mask], x=x[~mask])
    return d


def pctile_to_count(listings=None, featname=None):
    pctile = unpickle(PCTILE_DIR + '{}.pkl'.format(featname))
    pctile = pctile.reset_index().set_index('pctile').squeeze()
    s = listings[featname].rename('pctile')
    s = s.to_frame().join(pctile, on='pctile')[featname]
    return s


def slr_plots(data=None, listings=None, vals=None):
    # log seller feedback score
    score = pctile_to_count(listings=listings, featname=FDBK_SCORE)
    score[score == 0] = 1
    x = np.log10(score).values

    # store indicator
    store = listings[STORE].values

    # sale norm and sale indicator
    sale_norm = get_sale_norm(data[X_OFFER])
    is_sale = pd.Series(True, index=sale_norm.index)
    is_sale = is_sale.reindex(index=data[LOOKUP].index, fill_value=False)
    sale_norm = sale_norm.reindex(index=data[LOOKUP].index, fill_value=-99)

    # keys for dictionaries
    keys = ['Store', 'No store']

    # plots
    d = dict()
    d['slr_vals'] = two_ll(x=x, y=vals.values, mask=store, keys=keys)
    d['slr_sale'] = two_ll(x=x, y=is_sale.values, mask=store, keys=keys)
    d['slr_norm'] = two_ll(x=x, y=sale_norm.values, mask=store, keys=keys)

    return d


def main():
    # various data
    data = load_data(part=TEST)
    listings = load_feats('listings', lstgs=data[LOOKUP].index)
    vals = load_values(part=TEST, delta=DELTA_SHAP)

    # initialize output dictionary with seller plots
    print('Seller')
    d = slr_plots(data=data, listings=listings, vals=vals)  # type: dict

    # day of week of listing
    print('Day of week')
    d['coef_dowvals'] = dow_plot(listings=listings, vals=vals)

    # number of photos
    print('Photos')
    d['coef_photovals'] = photos_plot(listings=listings, vals=vals)

    # start price
    print('Start price')
    d['response_binvals'] = bin_plot(start_price=data[LOOKUP][START_PRICE],
                                     vals=vals)

    # pdf of values, for different values of delta
    print('Values distribution')
    kwargs = {'$\\delta = {}$'.format(delta):
              load_values(part=TEST, delta=delta)
              for delta in DELTA_CHOICES}
    d['pdf_values'] = kdens_wrapper(**kwargs)

    # save
    topickle(d, PLOT_DIR + 'values.pkl')


if __name__ == '__main__':
    main()

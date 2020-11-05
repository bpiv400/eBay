import numpy as np
import pandas as pd
from agent.util import load_values
from assess.util import kdens_wrapper, ll_wrapper, calculate_row
from processing.util import do_rounding, extract_day_feats, feat_to_pctile
from utils import topickle, load_data, load_feats
from agent.const import DELTA_CHOICES
from assess.const import DELTA_BYR, LOG10_BIN_DIM, LOG10_BO_DIM
from constants import PLOT_DIR, DAY
from featnames import LOOKUP, SLR_BO_CT, TEST, START_PRICE


def bin_plot(start_price=None, vals=None):
    x = start_price.values
    is_round, _ = do_rounding(x)
    x = np.log10(x)
    y = vals.values
    discrete = np.unique(x[is_round])
    line, dots, bw = ll_wrapper(y=y, x=x,
                                dim=LOG10_BIN_DIM,
                                discrete=discrete)
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
    photos = feat_to_pctile(listings['photos'], reverse=True, feat='photos')
    dim = range(int(photos.max() + 1))
    df = pd.DataFrame(index=dim, columns=['beta', 'err'])
    for i in dim:
        df.loc[i, :] = calculate_row(vals[photos == i])
    return df


def slr_plot(data=None, vals=None):
    assert np.all(data[LOOKUP].index == vals.index)
    x = np.log10(data[LOOKUP][SLR_BO_CT]).values  # log seller experience
    y = vals.values

    line, dots, bw = ll_wrapper(y=y, x=x,
                                discrete=[0],
                                dim=LOG10_BO_DIM)
    return line, dots


def main():
    # various data
    data = load_data(part=TEST)
    listings = load_feats('listings', lstgs=data[LOOKUP].index)
    vals = load_values(part=TEST, delta=DELTA_BYR)

    d = dict()

    # seller experience
    print('Seller')
    d['response_slrbo'] = slr_plot(data=data, vals=vals)

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

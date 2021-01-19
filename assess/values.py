import numpy as np
from agent.util import load_values
from assess.util import kdens_wrapper, ll_wrapper
from processing.util import do_rounding
from utils import unpickle, topickle, load_data
from agent.const import DELTA_CHOICES
from assess.const import LOG10_BIN_DIM, LOG10_BO_DIM, DELTA_ASSESS
from constants import PLOT_DIR, SIM_DIR
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
    print('bw: {}'.format(bw[0]))
    return line, dots


def slr_plot(data=None, y=None):
    x = np.log10(data[LOOKUP][SLR_BO_CT]).values  # log seller experience
    line, bw = ll_wrapper(y=y, x=x, dim=LOG10_BO_DIM)
    print('bw: {}'.format(bw[0]))
    return line


def main():
    # various data
    data = load_data(part=TEST)
    vals = load_values(part=TEST, delta=DELTA_ASSESS)

    d = dict()

    # seller experience
    print('Seller')
    d['response_slrbo'] = slr_plot(data=data, y=vals)
    df = unpickle(SIM_DIR + '{}/values.pkl'.format(TEST))
    assert np.all(df.index == data[LOOKUP].index)
    d['simple_slrbosale'] = slr_plot(data=data, y=df.p.values)
    d['simple_slrboprice'] = slr_plot(
        data=data, y=(df.x / data[LOOKUP][START_PRICE]).values)

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

import numpy as np
from agent.util import load_values
from assess.util import kdens_wrapper, ll_wrapper
from processing.util import do_rounding
from utils import unpickle, topickle, load_data, load_feats
from agent.const import DELTA_SLR
from assess.const import LOG10_BIN_DIM, LOG10_BO_DIM
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


def slr_plot(x=None, y=None):
    assert np.all(y.index == x.index)
    line, bw = ll_wrapper(y=y.values, x=x.values, dim=LOG10_BO_DIM)
    print('bw: {}'.format(bw[0]))
    return line


def main():
    # various data
    data = load_data()
    vals = load_values(delta=DELTA_SLR[-1])

    d = dict()

    # seller experience
    print('Seller')
    slrbo = load_feats('listings').loc[data[LOOKUP].index, SLR_BO_CT]
    x = np.log10(slrbo)  # log seller experience

    d['simple_slrbo'] = slr_plot(x=x, y=vals)
    df = unpickle(SIM_DIR + '{}/values.pkl'.format(TEST))
    assert np.all(df.index == data[LOOKUP].index)
    d['simple_slrbosale'] = slr_plot(x=x, y=df.p)
    d['simple_slrboprice'] = slr_plot(
        x=x, y=(df.x / data[LOOKUP][START_PRICE]))

    # start price
    print('Start price')
    d['response_binvals'] = bin_plot(start_price=data[LOOKUP][START_PRICE],
                                     vals=vals)

    # pdf of values, for different values of delta
    print('Values distribution')
    kwargs = {'$\\delta = {}$'.format(delta):
              load_values(part=TEST, delta=delta)
              for delta in DELTA_SLR}
    d['pdf_values'] = kdens_wrapper(**kwargs)

    # save
    topickle(d, PLOT_DIR + 'values.pkl')


if __name__ == '__main__':
    main()

import numpy as np
from agent.util import load_values
from analyze.util import kdens_wrapper, ll_wrapper, save_dict
from utils import unpickle, load_data, load_feats
from agent.const import DELTA_SLR
from analyze.const import LOG10_BO_DIM
from paths import SIM_DIR
from featnames import LOOKUP, SLR_BO_CT, TEST, START_PRICE


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

    # pdf of values, for different values of delta
    print('Values distribution')
    kwargs = {'$\\delta = {}$'.format(delta):
              load_values(part=TEST, delta=delta)
              for delta in DELTA_SLR}
    d['pdf_values'] = kdens_wrapper(**kwargs)

    # save
    save_dict(d, 'values')


if __name__ == '__main__':
    main()

import numpy as np
from agent.util import load_valid_data, only_byr_agent
from assess.util import ll_wrapper, kreg2
from utils import topickle, safe_reindex
from assess.const import NORM1_DIM_LONG, NORM1_BIN_MESH
from constants import PLOT_DIR
from featnames import X_OFFER, INDEX, NORM, LOOKUP, START_PRICE


def get_feats(data=None):
    norm = data[X_OFFER][NORM]
    norm1 = norm.xs(1, level=INDEX)
    norm1 = norm1[norm1 > .3]  # throw out small opening concessions
    # seller response
    norm2 = norm.xs(2, level=INDEX).reindex(index=norm1.index, fill_value=0)
    norm2 = 1 - norm2
    # log of start price
    log10_price = np.log10(safe_reindex(data[LOOKUP][START_PRICE],
                                        idx=norm1.index))
    return norm1.values, norm2.values, log10_price.values


def wrapper(x=None, dim=NORM1_DIM_LONG, bw=(.05,)):
    return lambda y: ll_wrapper(y=y, x=x, dim=dim, bw=bw)


def main():
    d = dict()

    # observed
    data = only_byr_agent(load_valid_data(byr=True))
    x, y, z = get_feats(data=data)
    f = wrapper(x=x)

    # norm2 ~ norm1
    d['simple_norm'], bw = f(y)

    # acc2 ~ norm1
    d['simple_accept'], bw = f((y == x))

    # rej2 ~ norm1
    d['simple_reject'], bw = f((y == 1))

    # # norm2 ~ norm1, list price
    # d['contour_normbin'], bw2 = kreg2(y=y, x1=x, x2=z,
    #                                   mesh=NORM1_BIN_MESH, bw=bw)
    # print('bin: {}'.format(bw2))

    topickle(d, PLOT_DIR + 'byr2.pkl')


if __name__ == '__main__':
    main()

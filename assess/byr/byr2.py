import numpy as np
from agent.util import load_valid_data, only_byr_agent
from assess.util import ll_wrapper, bin_plot
from utils import topickle, safe_reindex
from agent.const import COMMON_CONS
from assess.const import NORM1_DIM_LONG
from constants import PLOT_DIR
from featnames import X_OFFER, INDEX, TEST, NORM, LOOKUP, START_PRICE


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


def main():
    d = dict()

    # observed
    data = only_byr_agent(load_valid_data(part=TEST, byr=True))
    x, y, z = get_feats(data=data)

    # norm2 ~ norm1
    line, dots, bw = ll_wrapper(y, x,
                                dim=NORM1_DIM_LONG,
                                discrete=COMMON_CONS[1])
    d['response_norm'] = line, dots
    print('norm: {}'.format(bw[0]))

    # norm2 ~ norm1, list price
    d['contour_normbin'], bw2 = bin_plot(y=y, x1=x, x2=z)

    topickle(d, PLOT_DIR + 'byr2.pkl')


if __name__ == '__main__':
    main()

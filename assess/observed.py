import numpy as np
from assess.util import arrival_dist, hist_dist, delay_dist, con_dist, \
    norm_dist, ll_wrapper, continuous_cdf
from utils import topickle, load_data
from assess.const import LOG10_BO_DIM
from constants import PLOT_DIR, IDX
from featnames import TEST, X_THREAD, X_OFFER, LOOKUP, SLR_BO_CT, SLR, REJECT, \
    AUTO, EXP, INDEX, LSTG, START_PRICE, ACC_PRICE, DEC_PRICE


def main():
    data = load_data(part=TEST)

    d = dict()

    # offer distributions
    d['pdf_arrival'] = arrival_dist(data[X_THREAD])
    d['cdf_hist'] = hist_dist(data[X_THREAD])
    d['cdf_delay'] = delay_dist(data[X_OFFER])
    d['cdf_con'] = con_dist(data[X_OFFER])
    d['cdf_norm'] = norm_dist(data[X_OFFER])
    d['cdf_bin'] = continuous_cdf(data[LOOKUP][START_PRICE])
    d['cdf_autoacc'] = continuous_cdf(
        data[LOOKUP][ACC_PRICE] / data[LOOKUP][START_PRICE])
    d['cdf_autodec'] = continuous_cdf(
        data[LOOKUP][DEC_PRICE] / data[LOOKUP][START_PRICE])

    # expiration rate by seller experience
    slr_turn = data[X_OFFER].index.isin(IDX[SLR], level=INDEX)
    man_rej = (data[X_OFFER][REJECT] & ~data[X_OFFER][AUTO]).values
    exp = data[X_OFFER].loc[slr_turn & man_rej, EXP]
    bo_ct = data[LOOKUP][SLR_BO_CT].reindex(index=exp.index, level=LSTG)
    y, x = exp.values, np.log10(bo_ct).values

    line, dots, bw = ll_wrapper(y, x, dim=LOG10_BO_DIM, discrete=[0])
    d['response_expslrbo'] = line, dots

    # save
    topickle(d, PLOT_DIR + 'obs.pkl')


if __name__ == '__main__':
    main()

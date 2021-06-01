from agent.util import load_valid_data
from assess.util import ll_wrapper
from utils import topickle, safe_reindex
from assess.const import NORM1_DIM, NORM1_DIM_LONG
from constants import PLOT_DIR, HOUR
from featnames import X_OFFER, NORM, INDEX, LOOKUP, END_TIME, CLOCK


def main():
    d = dict()

    data = load_valid_data(byr=True, minimal=True, clock=True)

    # features
    norm1 = data[X_OFFER][NORM].xs(1, level=INDEX)
    norm2 = 1 - data[X_OFFER][NORM].xs(2, level=INDEX)
    norm2 = norm2.reindex(norm1.index)
    norm2.loc[norm2.isna()] = 1
    wide = data[CLOCK].unstack().loc[norm1.index][[1, 2]]
    idxna = wide[wide[2].isna()].index
    wide.loc[idxna, 2] = safe_reindex(data[LOOKUP][END_TIME], idx=idxna)
    hours = (wide[2] - wide[1]) / HOUR
    quick = hours <= 6

    # offer in response to first offer
    key = 'offer2norm'
    line, bw = ll_wrapper(y=norm2.values, x=norm1.values, dim=NORM1_DIM_LONG)
    print('{}: {}'.format(key, bw[0]))
    d['simple_{}'.format(key)] = line

    # response time
    key = 'offer2time'
    line, bw = ll_wrapper(y=quick.values, x=norm1.values, dim=NORM1_DIM)
    print('{}: {}'.format(key, bw[0]))
    d['simple_{}'.format(key)] = line

    topickle(d, PLOT_DIR + 'byrresponse.pkl')


if __name__ == '__main__':
    main()

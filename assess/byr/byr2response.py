import numpy as np
from agent.util import load_valid_data, only_byr_agent
from assess.util import ll_wrapper
from utils import topickle
from assess.const import NORM1_DIM
from constants import PLOT_DIR, HOUR
from featnames import X_OFFER, INDEX, TEST, NORM, CLOCK, AUTO, REJECT


def get_feats(data=None):
    df = data[X_OFFER][[NORM, AUTO, REJECT]]
    norm1 = df[NORM].xs(1, level=INDEX)
    norm1 = norm1[norm1 > .33]  # throw out small opening concessions

    feats = dict()
    wide = data[CLOCK].unstack().loc[norm1.index]
    tdiff = (wide[2] - wide[1]) / HOUR
    feats['responds'] = (norm1, ~tdiff.isna())
    autorej = (df[REJECT] & df[AUTO]).xs(2, level=INDEX).reindex(
        index=norm1.index, fill_value=False)
    feats['autorej'] = (norm1, autorej)
    autoacc = (~df[REJECT] & df[AUTO]).xs(2, level=INDEX).reindex(
        index=norm1.index, fill_value=False)
    feats['autoacc'] = (norm1, autoacc)
    hours = tdiff[~tdiff.isna()]
    feats['hours'] = (norm1.loc[hours.index], hours)
    manhours = tdiff[tdiff > 0]
    feats['manhours'] = (norm1.loc[manhours.index], manhours)

    for k, v in feats.items():
        assert np.all(v[0].index == v[1].index)
        feats[k] = (v[0].values, v[1].values)

    return feats


def main():
    d = dict()

    # observed
    data = only_byr_agent(load_valid_data(part=TEST, byr=True))
    feats = get_feats(data=data)

    # norm2 ~ response type
    for k, v in feats.items():
        d['simple_{}'.format(k)], bw = ll_wrapper(v[1], v[0], dim=NORM1_DIM)
        print('{}: {}'.format(k, bw[0]))

    topickle(d, PLOT_DIR + 'byr2response.pkl')


if __name__ == '__main__':
    main()

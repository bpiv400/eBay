import pandas as pd
from agent.util import find_best_run, load_valid_data
from assess.util import ll_wrapper
from utils import topickle, load_data
from agent.const import AGENT_CONS, DELTA_CHOICES
from assess.const import NORM1_DIM
from constants import PLOT_DIR
from featnames import X_OFFER, CON, INDEX, TEST, AUTO, NORM, ACCEPT, REJECT


def get_y_x(norm1=None, norm2=None, con2=None, key=None):
    if key == ACCEPT:
        y, x = con2.values == 1, norm1.values
    elif key == REJECT:
        y, x = con2.values == 0, norm1.values
    elif key == NORM:
        y, x = norm2.values, norm1.values
    elif key == CON:
        is_counter = (con2 < 1) & (con2 > 0)
        y, x = con2[is_counter].values, norm1[is_counter].values
    else:
        raise NotImplementedError('Invalid key: {}'.format(key))
    return y, x


def main():
    d, bw = dict(), dict()

    # load data
    data = load_data(part=TEST)

    # first offer and response
    df = data[X_OFFER][[CON, NORM]]
    norm1 = df[NORM].xs(1, level=INDEX)
    norm1 = norm1[(.33 <= norm1) & (norm1 < 1)]  # remove BINs and very small offers
    norm2 = 1 - df[NORM].xs(2, level=INDEX).reindex(index=norm1.index, fill_value=0)
    norm2 = norm2.reindex(index=norm1.index)
    con2 = df[CON].xs(2, level=INDEX).reindex(index=norm1.index, fill_value=0)

    for key in [ACCEPT, REJECT, CON, NORM]:
        y, x = get_y_x(norm1=norm1, norm2=norm2, con2=con2, key=key)
        line, dots, bw[key] = ll_wrapper(y, x,
                                         dim=NORM1_DIM,
                                         discrete=AGENT_CONS[1])
        line.columns = pd.MultiIndex.from_product([['Data'], line.columns])
        dots.columns = pd.MultiIndex.from_product([['Data'], dots.columns])
        d['response_{}'.format(key)] = line, dots
        print('{}: {}'.format(key, bw[key][0]))

    # seller runs
    for delta in DELTA_CHOICES:
        run_dir = find_best_run(byr=False, delta=delta)
        if run_dir is None:
            continue
        data = load_valid_data(part=TEST, run_dir=run_dir)

        df = data[X_OFFER].loc[~data[X_OFFER][AUTO], [CON, NORM]]
        norm2 = 1 - df[NORM].xs(2, level=INDEX)
        norm1 = df[NORM].xs(1, level=INDEX).reindex(index=norm2.index)
        norm1 = norm1[norm1 > .33]
        norm2 = norm2.reindex(index=norm1.index)
        con2 = df[CON].xs(2, level=INDEX).reindex(index=norm1.index)

        for key in [ACCEPT, REJECT, CON, NORM]:
            y, x = get_y_x(norm1=norm1, norm2=norm2, con2=con2, key=key)
            line, dots, _ = ll_wrapper(y, x,
                                       dim=NORM1_DIM,
                                       discrete=AGENT_CONS[1],
                                       bw=bw[key],
                                       ci=False)
            level = 'Agent_{}'.format(delta)
            d['response_{}'.format(key)][0].loc[:, (level, 'beta')] = line
            d['response_{}'.format(key)][1].loc[:, (level, 'beta')] = dots

    topickle(d, PLOT_DIR + 'turn2.pkl')


if __name__ == '__main__':
    main()

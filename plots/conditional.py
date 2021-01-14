from compress_pickle import load
import pandas as pd
from plots.util import grouped_bar
from plots.const import SPLIT_LABELS, SPLIT_YLABELS
from constants import PLOT_DIR
from featnames import SIM, OBS


def construct_df(split_feat, ps, key):
    # lists of outcomes
    bin_sim = [ps[i][SIM][key] for i in range(len(ps))]
    bin_obs = [ps[i][OBS][key] for i in range(len(ps))]
    # combine into dataframe
    df = pd.DataFrame(list(zip(bin_sim, bin_obs)),
                      columns=[SIM, OBS],
                      index=SPLIT_LABELS[split_feat])
    # percentage points
    df *= 100
    return df


def sale_rate(split_feat, ps):
    name = 'sale_{}'.format(split_feat)
    print(name)

    df = construct_df(split_feat, ps, 'sale')

    grouped_bar(name, df,
                horizontal=True,
                xlim=[0, 100],
                ylabel=SPLIT_YLABELS[split_feat],
                xlabel='% of listing windows that end in a sale')


def sale_price(split_feat, ps):
    name = 'price_{}'.format(split_feat)
    print(name)

    df = construct_df(split_feat, ps, 'price')

    grouped_bar(name, df,
                horizontal=True,
                xlim=[70, 100],
                ylabel=SPLIT_YLABELS[split_feat],
                xlabel='Average sale price, as % of buy-it-now price')


def main():
    for split_feat in SPLIT_LABELS.keys():
        # load list of dictionaries
        ps = load(PLOT_DIR + 'p_{}.pkl'.format(split_feat))

        # sale rate plot
        sale_rate(split_feat, ps)

        # average sale price plot
        sale_price(split_feat, ps)


if __name__ == '__main__':
    main()

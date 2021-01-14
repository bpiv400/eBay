import pandas as pd
from agent.util import only_byr_agent, load_valid_data, get_run_dir
from assess.util import discrete_pdf
from utils import topickle
from agent.const import TURN_COST_CHOICES, DELTA_BYR
from constants import PLOT_DIR
from featnames import TEST, X_OFFER, CON, INDEX


def get_sale_turn_pdf(con):
    sale_turn = con.loc[con == 1].index
    s = pd.Series(index=sale_turn).reset_index(INDEX)[INDEX].squeeze()
    return discrete_pdf(s)


def main():
    d = dict()

    data = only_byr_agent(load_valid_data(part=TEST, byr=True))
    con = data[X_OFFER][CON]

    df = get_sale_turn_pdf(con).rename('Humans')

    for turn_cost in TURN_COST_CHOICES:
        run_dir = get_run_dir(byr=True,
                              delta=DELTA_BYR[-1],
                              turn_cost=turn_cost)
        data = load_valid_data(part=TEST, run_dir=run_dir, byr=True)
        if data is None:
            continue
        data = only_byr_agent(data)
        con = data[X_OFFER][CON]

        col = get_sale_turn_pdf(con).rename('${}'.format(turn_cost))
        col[col.isna()] = 0.
        df = pd.concat([df, col], axis=1)

    d['bar_saleturn'] = df

    topickle(d, PLOT_DIR + 'byrcost.pkl')


if __name__ == "__main__":
    main()

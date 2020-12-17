import argparse
import pandas as pd
from assess.util import get_last, estimate_tree
from agent.util import get_run_dir, load_valid_data
from utils import compose_args
from agent.const import AGENT_PARAMS
from constants import DAY, MAX_DAYS
from featnames import LOOKUP, AUTO, CON, NORM, START_PRICE, START_TIME, \
    BYR_HIST, X_OFFER, X_THREAD, INDEX, CLOCK, THREAD, TEST, BYR


def main():
    # agent params from command line
    parser = argparse.ArgumentParser()
    compose_args(arg_dict=AGENT_PARAMS, parser=parser)
    params = vars(parser.parse_args())
    assert not params[BYR]

    run_dir = get_run_dir(**params)
    data = load_valid_data(part=TEST, run_dir=run_dir, byr=False)

    for turn in [2, 4, 6]:
        print('Turn {}'.format(turn))

        # find valid indices
        is_turn = data[X_OFFER].index.get_level_values(INDEX) == turn
        idx = data[X_OFFER][~data[X_OFFER][AUTO] & is_turn].index

        # outcome
        con = data[X_OFFER].loc[idx, CON]
        y = pd.Series('', index=con.index)
        y.loc[con == 0] = 'Reject'
        y.loc[con == 1] = 'Accept'
        y.loc[con == .5] = 'Half'
        y.loc[(con > 0) & (con < .5)] = 'Low'
        y.loc[(con > .5) & (con < 1)] = 'High'
        print(list(y.unique()))

        con_rate = con.groupby(con).count() / len(con)
        print(con_rate)

        # features
        X = data[X_THREAD][BYR_HIST].reindex(index=idx).to_frame()
        X['thread_num'] = X.index.get_level_values(THREAD)
        tdiff = data[CLOCK].loc[idx] - data[LOOKUP][START_TIME]
        X['elapsed'] = tdiff / (MAX_DAYS * DAY)
        X = X.join(get_last(data[X_OFFER][NORM]))
        X = X.join(data[LOOKUP][START_PRICE])

        # decision tree
        estimate_tree(X=X, y=y, max_depth=2)


if __name__ == '__main__':
    main()

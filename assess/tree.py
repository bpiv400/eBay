import argparse
from sklearn.tree import DecisionTreeClassifier, export_text
from assess.util import get_last
from agent.util import find_best_run
from utils import load_data
from agent.const import DELTA_CHOICES
from constants import DAY, MAX_DAYS
from featnames import LOOKUP, AUTO, CON, NORM, START_PRICE, START_TIME, \
    BYR_HIST, X_OFFER, X_THREAD, INDEX, CLOCK, THREAD, TEST


def main():
    # delta from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta', type=float,
                        choices=DELTA_CHOICES, required=True)
    delta = parser.parse_args().delta

    run_dir = find_best_run(byr=False, delta=delta)
    data = load_data(part=TEST, run_dir=run_dir)
    # data = load_data(part=TEST)

    for turn in [2, 4, 6]:
        print('Turn {}'.format(turn))

        # find valid indices
        is_turn = data[X_OFFER].index.get_level_values(INDEX) == turn
        idx = data[X_OFFER][~data[X_OFFER][AUTO] & is_turn].index

        # outcome
        con = data[X_OFFER].loc[idx, CON]
        y = (((0 < con) & (con < 1)) + 2 * (con == 1)).values

        con_rate = con.groupby(con).count() / len(con)
        con_rate = con_rate[con_rate > .01]
        print(con_rate)

        # features
        X = data[X_THREAD][BYR_HIST].reindex(index=idx).to_frame()
        X['thread_num'] = X.index.get_level_values(THREAD)
        tdiff = data[CLOCK].loc[idx] - data[LOOKUP][START_TIME]
        X['elapsed'] = tdiff / (MAX_DAYS * DAY)
        X = X.join(get_last(data[X_OFFER][NORM]))
        X = X.join(data[LOOKUP][START_PRICE])

        # split out columns names
        cols = list(X.columns)
        X = X.values

        # decision tree
        clf = DecisionTreeClassifier(max_depth=1).fit(X, y)
        r = export_text(clf, feature_names=cols)
        print(r)


if __name__ == '__main__':
    main()

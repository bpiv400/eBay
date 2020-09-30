from sklearn.tree import DecisionTreeClassifier, export_text
from assess.util import get_last
from agent.util import find_best_run
from utils import load_data
from constants import TEST, DAY, MAX_DAYS
from featnames import LOOKUP, AUTO, CON, NORM, START_PRICE, START_TIME, \
    BYR_HIST, X_OFFER, X_THREAD, INDEX, CLOCK, THREAD


def main():
    run_dir = find_best_run(byr=False, delta=.75)
    data = load_data(part=TEST, run_dir=run_dir)

    for turn in [2, 4, 6]:
        # find valid indices
        is_turn = data[X_OFFER].index.get_level_values(INDEX) == turn
        idx = data[X_OFFER][~data[X_OFFER][AUTO] & is_turn].index

        # outcome
        con = data[X_OFFER].loc[idx, CON]
        # y = ((0 < con) & (con < 1)) + 2 * (con == 1)
        y = (con == 1).values
        print('Turn {0:d} accept rate: {1:2.1f}%'.format(
            turn, 100 * y.mean()))
        # print('Turn {0:d} concession rate: {1:2.1f}%'.format(
        #     turn, 100 * (y == 1).mean()))
        # print('Turn {0:d} accept rate: {1:2.1f}%'.format(
        #     turn, 100 * (y == 2).mean()))

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
        clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
        r = export_text(clf, feature_names=cols)
        print(r)


if __name__ == '__main__':
    main()

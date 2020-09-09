from sklearn.tree import DecisionTreeClassifier, export_text
from assess.util import load_data, get_last_norm, get_log_dir
from agent.util import find_best_run
from utils import load_file
from constants import TEST, DAY
from featnames import LOOKUP, AUTO, CON, NORM, START_PRICE, START_TIME, BYR_HIST


def main():
    lookup = load_file(TEST, LOOKUP)
    # run_dir = find_best_run()
    run_dir = get_log_dir(byr=False) + 'run_nocon_0_1/'
    data = load_data(part=TEST, run_dir=run_dir)
    threads, offers, clock = [data[k] for k in ['threads', 'offers', 'clock']]

    for turn in [2, 4, 6]:
        # find valid indices
        is_turn = offers.index.get_level_values('index') == turn
        idx = offers[~offers[AUTO] & is_turn].index

        # outcome
        con = offers.loc[idx, CON]
        # y = ((0 < con) & (con < 1)) + 2 * (con == 1)
        y = con == 1
        y = y.values
        print('Turn {0:d} accept rate: {1:2.1f}%'.format(
            turn, 100 * y.mean()))
        # print('Turn {0:d} concession rate: {1:2.1f}%'.format(
        #     turn, 100 * (y == 1).mean()))
        # print('Turn {0:d} accept rate: {1:2.1f}%'.format(
        #     turn, 100 * (y == 2).mean()))

        # features
        X = threads[BYR_HIST].reindex(index=idx).to_frame()
        X['thread_num'] = X.index.get_level_values('thread')
        X['days'] = (clock.loc[idx] - lookup[START_TIME]) / DAY
        X = X.join(get_last_norm(offers[NORM]))
        X = X.join(lookup[START_PRICE])

        # split out columns names
        cols = list(X.columns)
        X = X.values

        # decision tree
        clf = DecisionTreeClassifier(max_depth=2).fit(X, y)
        r = export_text(clf, feature_names=cols)
        print(r)


if __name__ == '__main__':
    main()

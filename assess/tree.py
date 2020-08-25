from sklearn.tree import DecisionTreeClassifier, export_text
from assess.util import find_best_run, load_data, get_last_norm
from utils import load_file
from constants import TEST, WEEK
from featnames import LOOKUP, AUTO, CON, NORM, START_PRICE, START_TIME, BYR_HIST


def main():
    lookup = load_file(TEST, LOOKUP)
    data = load_data(part=TEST, run_dir=find_best_run())
    threads, offers, clock = [data[k] for k in ['threads', 'offers', 'clock']]

    for turn in [2, 4, 6]:
        # find valid indices
        is_turn = offers.index.get_level_values('index') == turn
        idx = offers[~offers[AUTO] & is_turn].index

        # outcome
        y = (offers.loc[idx, CON].astype('int64') == 1).values
        print('Turn {0:d} baserate: {1:2.1f}%'.format(turn, 100 * y.mean()))

        # features
        X = threads[BYR_HIST].reindex(index=idx).to_frame()
        X['thread_num'] = X.index.get_level_values('thread')
        X['months'] = (clock.loc[idx] - lookup[START_TIME]) / WEEK
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

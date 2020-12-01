from agent.util import get_log_dir, find_best_run
from utils import unpickle, topickle
from agent.const import DELTA_SLR
from constants import PLOT_DIR
from featnames import TEST, OBS


def main():
    d = dict()

    # seller
    for delta in DELTA_SLR:
        run_id = find_best_run(byr=False, delta=delta).split('/')[-2]
        log_dir = get_log_dir(byr=False, delta=delta)
        path = log_dir + '{}.pkl'.format(TEST)
        df = unpickle(path)[['norm', 'dollar']]

        # rename rows
        newkeys = {OBS: 'Data', 'heuristic': 'Heuristic', run_id: 'Agent'}
        df = df.rename(index=newkeys)
        df = df.loc[newkeys.values(), :]

        for c in df.columns:
            d['bar_{}_{}'.format(c, delta)] = df[c]

    topickle(d, PLOT_DIR + 'evaluate.pkl')


if __name__ == '__main__':
    main()

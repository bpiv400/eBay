from agent.util import get_log_dir, find_best_run
from utils import unpickle, topickle
from assess.const import DELTA_SLR
from constants import PLOT_DIR
from featnames import TEST, NORM, OBS


def main():
    d = dict()

    # seller
    run_id = find_best_run(byr=False, delta=DELTA_SLR).split('/')[-2]
    log_dir = get_log_dir(byr=False, delta=DELTA_SLR)
    path = log_dir + '{}.pkl'.format(TEST)
    df = unpickle(path)

    # restrict index
    newkeys = {OBS: 'Data', 'heuristic': 'Heuristic', run_id: 'Agent'}
    df = df.loc[newkeys.keys(), :]
    df = df.rename(index=newkeys)

    d['bar_slrnorm'] = df[NORM]
    d['bar_slrdollar'] = df['dollar']

    topickle(d, PLOT_DIR + 'evaluate.pkl')


if __name__ == '__main__':
    main()

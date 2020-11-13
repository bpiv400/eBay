from agent.util import get_log_dir, find_best_run
from utils import unpickle, topickle
from agent.const import DELTA_CHOICES
from constants import PLOT_DIR
from featnames import TEST, NORM, OBS


def select_cols(df=None, run_id=None, key=None):
    newkeys = {OBS: 'Data', run_id: 'Agent'}
    if key == 'heuristic':
        newkeys[key] = 'Heuristic'
    elif key == 'slrrej':
        newkeys[key] = 'Turn 2 reject'
    df = df.loc[newkeys.keys(), :]
    df = df.rename(index=newkeys)
    return df[NORM], df['dollar']


def main():
    d = dict()

    # seller
    for delta in DELTA_CHOICES:
        run_id = find_best_run(byr=False, delta=delta).split('/')[-2]
        log_dir = get_log_dir(byr=False, delta=delta)
        path = log_dir + '{}.pkl'.format(TEST)
        df = unpickle(path)

        for name in ['heuristic', 'slrrej']:
            suf = '{}_{}'.format(name, delta)
            d['bar_norm_{}'.format(suf)], d['bar_dollar_{}'.format(suf)] = \
                select_cols(df=df, run_id=run_id, key=name)

    topickle(d, PLOT_DIR + 'evaluate.pkl')


if __name__ == '__main__':
    main()

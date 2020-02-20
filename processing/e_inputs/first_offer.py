import argparse
import numpy as np
import pandas as pd
from processing.processing_utils import load_file, get_x_thread, \
    init_x, save_files
from constants import PARTITIONS
from featnames import CON, MSG, AUTO, EXP


# loads data and calls helper functions to construct train inputs
def process_inputs(part, outcome, role):
    # load dataframes
    offers = load_file(part, 'x_offer')
    threads = load_file(part, 'x_thread')

    # outcome and master index
    df = offers.xs(1, level='index')
    assert all(~df[AUTO]) and all(~df[EXP])
    if outcome == CON:
        y = (df[CON] * 100).astype('int8')
    elif outcome == MSG:
        y = df.loc[df[CON] < 1, MSG]
    idx = y.index

    # listing features
    x = init_x(part, idx, drop_slr=True)

    # thread features
    x_thread = get_x_thread(threads, idx)
    x['lstg'] = pd.concat([x['lstg'], x_thread], axis=1)

    return {'y': y, 'x': x}


def main():
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str)
    parser.add_argument('--outcome', type=str)
    args = parser.parse_args()
    part, outcome = args.part, args.outcome
    assert part in PARTITIONS
    assert outcome in [CON, MSG]
    name = 'first_%s' % outcome
    print('%s/%s' % (part, name))

    # input dataframes, output processed dataframes
    d = process_inputs(part, outcome, role)

    # save various output files
    save_files(d, part, name)


if __name__ == '__main__':
    main()

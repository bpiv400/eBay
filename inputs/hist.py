import numpy as np
from inputs.util import save_files, get_ind_x
from utils import load_file, input_partition, load_feats
from featnames import CLOCK_FEATS, THREAD_COUNT, BYR_HIST, LOOKUP, THREAD, \
    X_THREAD, X_OFFER, INDEX, BYR_HIST_MODEL


def process_inputs(part):
    lstgs = load_file(part, LOOKUP).index
    threads = load_file(part, X_THREAD)
    offers = load_file(part, X_OFFER)

    # thread features
    clock_feats = offers.xs(1, level=INDEX)[CLOCK_FEATS]
    x_thread = clock_feats.join(threads.drop(BYR_HIST, axis=1))
    x_thread[THREAD_COUNT] = x_thread.index.get_level_values(level=THREAD) - 1
    x = {THREAD: x_thread}

    # outcome
    y = load_feats('threads', lstgs=lstgs)[BYR_HIST]
    assert np.all(y.index == x[THREAD].index)

    # indices for listing features
    idx_x = get_ind_x(lstgs=lstgs, idx=y.index)

    return {'y': y, 'x': x, 'idx_x': idx_x}


def main():
    # partition name from command line
    part = input_partition()
    print('{}/{}'.format(part, BYR_HIST_MODEL))

    # input dataframes, output processed dataframes
    d = process_inputs(part)

    # save various output files
    save_files(d, part, BYR_HIST_MODEL)


if __name__ == '__main__':
    main()

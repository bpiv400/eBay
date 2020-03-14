import argparse
import pandas as pd
from processing.processing_utils import load_file, get_x_thread, \
    get_x_offer, init_x, save_files, get_y_con, check_zero, calculate_remaining
from constants import IDX, BYR_PREFIX, SLR_PREFIX
from featnames import CON, INT_REMAINING


# loads data and calls helper functions to construct train inputs
def process_inputs(part, role):
    # load dataframes
    offers = load_file(part, 'x_offer')
    threads = load_file(part, 'x_thread')

    # outcome and master index
    df = offers[offers.index.isin(IDX[role], level='index')]
    y = get_y_con(df)
    idx = y.index

    # listing features
    x = init_x(part, idx)

    # thread features
    x_thread = get_x_thread(threads, idx)
    x_thread[INT_REMAINING] = calculate_remaining(part, idx)
    x['lstg'] = pd.concat([x['lstg'], x_thread], axis=1)

    # offer features
    x.update(get_x_offer(offers, idx, outcome=CON, role=role))

    # error checking
    check_zero(x)

    return {'y': y, 'x': x}


def main():
    # extract parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str)
    parser.add_argument('--role', type=str)
    parser.add_argument('--delay', action='store_true')
    args = parser.parse_args()
    part, role, delay = args.part, args.role, args.delay
    assert role in [BYR_PREFIX, SLR_PREFIX]
    if delay:
        raise NotImplementedError()
    else:
        name = 'init_{}'.format(role)
    print('%s/%s' % (part, name))

    # input dataframes, output processed dataframes
    d = process_inputs(part, role)

    # save various output files
    save_files(d, part, name)


if __name__ == '__main__':
    main()

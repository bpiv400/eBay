from processing.processing_utils import input_partition, load_file, \
    get_arrival_times, save_files, init_x
from processing.processing_consts import INTERVAL, INTERVAL_COUNTS
from constants import ARRIVAL_PREFIX


def process_inputs(part):
    # load timestamps
    lstg_start = load_file(part, 'lookup').start_time
    thread_start = load_file(part, 'clock').xs(1, level='index')

    # first arrival time
    clock = get_arrival_times(lstg_start, thread_start)
    clock = clock[clock.index.isin([0, 1], level='thread')]
    clock = clock.sort_index().unstack()
    diff = clock[1] - clock[0]

    # interarrival time in periods
    y = diff // INTERVAL[ARRIVAL_PREFIX]
    y[y.isna()] = INTERVAL_COUNTS[ARRIVAL_PREFIX]
    y = y.astype('int64')
    idx = y.index

    # listing features
    x = init_x(part, idx)

    return {'y': y, 'x': x}


def main():
    # partition name from command line
    part = input_partition()
    print('%s/first_arrival' % part)

    # create input dictionary
    d = process_inputs(part)

    # save various output files
    save_files(d, part, 'first_arrival')


if __name__ == '__main__':
    main()

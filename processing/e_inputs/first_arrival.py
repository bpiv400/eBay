from processing.processing_utils import input_partition, load_file, init_x
from processing.e_inputs.inputs_utils import get_arrival_times, save_files
from processing.processing_consts import INTERVAL, INTERVAL_COUNTS
from constants import FIRST_ARRIVAL_MODEL


def get_first_arrival_period(lstg_start, thread_start):
    # first arrival time
    clock = get_arrival_times(lstg_start, thread_start)
    clock = clock[clock.index.isin([0, 1], level='thread')]
    clock = clock.sort_index().unstack()
    diff = clock[1] - clock[0]

    # interarrival time in periods
    y = diff // INTERVAL[1]
    y[y.isna()] = INTERVAL_COUNTS[1]
    y = y.astype('int64')

    return y


def process_inputs(part):
    # load timestamps
    lstg_start = load_file(part, 'lookup').start_time
    thread_start = load_file(part, 'clock').xs(1, level='index')

    # arrival period
    y = get_first_arrival_period(lstg_start, thread_start)
    idx = y.index

    # listing features
    x = init_x(part, idx)

    return {'y': y, 'x': x}


def main():
    # partition name from command line
    part = input_partition()
    print('{}/{}'.format(part, FIRST_ARRIVAL_MODEL))

    # create input dictionary
    d = process_inputs(part)

    # save various output files
    save_files(d, part, FIRST_ARRIVAL_MODEL)


if __name__ == '__main__':
    main()

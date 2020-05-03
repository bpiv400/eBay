from processing.processing_utils import input_partition, init_x, load_file
from processing.e_inputs.inputs_utils import get_arrival_times, save_files
from processing.processing_consts import INTERVAL, INTERVAL_COUNTS
from constants import FIRST_ARRIVAL_MODEL


def process_inputs(part):
    # data
    clock = load_file(part, 'clock')
    lstg_start = load_file(part, 'lookup').start_time
    lstg_end = load_file(part, 'lstg_end')

    # first arrival time
    arrivals = get_arrival_times(clock, lstg_start, lstg_end,
                                 append_last=False)
    arrivals = arrivals[arrivals.index.isin([0, 1], level='thread')]
    arrivals = arrivals.sort_index().unstack()
    diff = arrivals[1] - arrivals[0]

    # interarrival time in periods
    y = diff // INTERVAL[1]
    y[y.isna()] = INTERVAL_COUNTS[1]
    y = y.astype('int64')

    # listing features
    x = init_x(part, y.index)

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

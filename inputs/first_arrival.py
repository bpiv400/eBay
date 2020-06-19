from inputs.util import get_arrival_times, save_files
from utils import input_partition, load_file, init_x
from inputs.const import ARRIVAL_INTERVAL, ARRIVAL_INTERVAL_CT
from constants import FIRST_ARRIVAL_MODEL, HOUR
from featnames import START_TIME


def process_inputs(part):
    # data
    clock = load_file(part, 'clock')
    lstg_start = load_file(part, 'lookup')[START_TIME]
    lstg_end = load_file(part, 'lstg_end')

    # first arrival time
    arrivals = get_arrival_times(clock, lstg_start, lstg_end,
                                 append_last=False)
    arrivals = arrivals[arrivals.index.isin([0, 1], level='thread')]
    arrivals = arrivals.sort_index().unstack()
    diff = arrivals[1] - arrivals[0]

    # interarrival time in periods
    y = diff // ARRIVAL_INTERVAL

    # fill in missings
    idx0 = y[y.isna()].index
    hours = ((lstg_end - lstg_start + 1) / HOUR)[idx0]
    y[idx0] = hours - (ARRIVAL_INTERVAL_CT + 1)
    y = y.astype(clock.dtype)

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

from inputs.util import get_arrival_times, save_files, get_ind_x
from utils import input_partition, load_file
from featnames import START_TIME, END_TIME, LOOKUP
from constants import FIRST_ARRIVAL_MODEL, INTERVAL_ARRIVAL


def process_inputs(part):
    # data
    clock = load_file(part, 'clock')
    lookup = load_file(part, LOOKUP)
    lstg_start = lookup[START_TIME]
    lstg_end = lookup[END_TIME]

    # first arrival time
    arrivals = get_arrival_times(clock=clock,
                                 lstg_start=lstg_start,
                                 lstg_end=lstg_end,
                                 append_last=False)
    arrivals = arrivals[arrivals.index.isin([0, 1], level='thread')]
    arrivals = arrivals.sort_index().unstack()
    diff = arrivals[1] - arrivals[0]

    # interarrival time in periods
    y = diff // INTERVAL_ARRIVAL
    assert not y.isna().all()

    # indices for listing features
    idx_x = get_ind_x(lstgs=lookup.index, idx=y.index)

    return {'y': y, 'idx_x': idx_x}


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

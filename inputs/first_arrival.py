from inputs.util import get_arrival_times, save_files, get_ind_x
from utils import input_partition, load_file
from featnames import START_TIME, END_TIME
from constants import FIRST_ARRIVAL_MODEL, HOUR, INTERVAL_ARRIVAL, INTERVAL_CT_ARRIVAL

AGENT = False


def process_inputs(part):
    # data
    clock = load_file(part, 'clock', agent=AGENT)
    lookup = load_file(part, 'lookup', agent=AGENT)
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

    # fill in missings
    idx0 = y[y.isna()].index
    hours = ((lstg_end - lstg_start + 1) / HOUR)[idx0]
    y[idx0] = hours - (INTERVAL_CT_ARRIVAL + 1)
    y = y.astype(clock.dtype)

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

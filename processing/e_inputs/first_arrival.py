from processing.processing_utils import input_partition, init_x, \
    get_obs_outcomes
from processing.e_inputs.inputs_utils import get_arrival_times, save_files
from processing.processing_consts import INTERVAL, INTERVAL_COUNTS
from constants import FIRST_ARRIVAL_MODEL


def process_inputs(d, part):
    # first arrival time
    clock = get_arrival_times(d, append_last=False)
    clock = clock[clock.index.isin([0, 1], level='thread')]
    clock = clock.sort_index().unstack()
    diff = clock[1] - clock[0]

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

    # dictionary of components
    obs = get_obs_outcomes(part, timestamps=True)

    # create input dictionary
    d = process_inputs(obs, part)

    # save various output files
    save_files(d, part, FIRST_ARRIVAL_MODEL)


if __name__ == '__main__':
    main()

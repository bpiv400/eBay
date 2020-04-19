from processing.e_inputs.hist import process_inputs
from processing.f_discrim.discrim_utils import create_arrival_discrim_input
from constants import BYR_HIST_MODEL, HIST_QUANTILES
from featnames import BYR_HIST


def construct_y(y):
    return (y / HIST_QUANTILES).rename(BYR_HIST).to_frame()


def main():
    create_arrival_discrim_input(BYR_HIST_MODEL,
                                 process_inputs,
                                 construct_y)


if __name__ == '__main__':
    main()

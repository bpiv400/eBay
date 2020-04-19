import pandas as pd
from processing.e_inputs.first_arrival import process_inputs
from processing.f_discrim.discrim_utils import create_arrival_discrim_input
from processing.processing_consts import INTERVAL_COUNTS
from constants import FIRST_ARRIVAL_MODEL
from featnames import MONTHS_SINCE_LSTG


def construct_y(y):
    y1 = (y / INTERVAL_COUNTS[1]).rename(MONTHS_SINCE_LSTG)
    y2 = (y == 1).rename('no_arrival')
    df = pd.concat([y1, y2], axis=1)
    return df


def main():
    create_arrival_discrim_input(FIRST_ARRIVAL_MODEL,
                                 process_inputs,
                                 construct_y)


if __name__ == '__main__':
    main()

from compress_pickle import dump
import pandas as pd
from utils import get_model_predictions, input_partition, load_file
from constants import FIRST_ARRIVAL_MODEL, PARTS_DIR


def main():
    # partition
    part = input_partition()

    # lookup file
    lookup = load_file(part, 'lookup')

    # load data as dictionary of 32-bit numpy arrays
    x = load_file(part, 'x_lstg')
    x = {k: v.astype('float32').values for k, v in x.items()}

    # model predictions
    p = get_model_predictions(FIRST_ARRIVAL_MODEL, x)

    # put in series
    p0 = pd.Series(p[:,-1], index=lookup.index, name='p_no_arrival')

    # join with lookup
    lookup = lookup.join(p0)

    # save
    dump(lookup, PARTS_DIR + '{}/lookup.gz'.format(part))


if __name__ == '__main__':
    main()

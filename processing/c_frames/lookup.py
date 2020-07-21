from compress_pickle import dump
import pandas as pd
from constants import PARTS_DIR
from featnames import LOOKUP, META, START_TIME, END_TIME, START_PRICE, \
    DEC_PRICE, ACC_PRICE, START_DATE
from processing.util import get_lstgs, load_feats
from utils import input_partition


def create_lookup(lstgs=None):
    # load data
    listings = load_feats('listings', lstgs=lstgs)

    # start time instead of start date
    start_time = listings[START_DATE].astype('int64') * 24 * 3600
    start_time = start_time.rename(START_TIME)

    # subset features
    lookup = listings[[META, START_PRICE, DEC_PRICE, ACC_PRICE]]
    lookup = pd.concat([lookup, start_time, listings[END_TIME]], axis=1)

    return lookup


def main():
    part = input_partition()
    print('{}/lookup'.format(part))

    lookup = create_lookup(lstgs=get_lstgs(part))
    dump(lookup, PARTS_DIR + '{}/{}.gz'.format(part, LOOKUP))


if __name__ == "__main__":
    main()

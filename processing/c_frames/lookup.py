import pandas as pd
from constants import PARTS_DIR, DAY
from featnames import LOOKUP, META, START_TIME, END_TIME, START_PRICE, \
    DEC_PRICE, START_DATE, SLR_BO_CT, STORE
from processing.util import get_lstgs, load_feats
from utils import topickle, input_partition


def create_lookup(lstgs=None):
    # load data
    listings = load_feats('listings', lstgs=lstgs)

    # start time instead of start date
    start_time = listings[START_DATE].astype('int64') * DAY
    start_time = start_time.rename(START_TIME)

    # subset features
    lookup = listings[[META, START_PRICE, DEC_PRICE, SLR_BO_CT, STORE]]
    lookup = pd.concat([lookup, start_time, listings[END_TIME]], axis=1)

    return lookup


def main():
    part = input_partition()
    print('{}/lookup'.format(part))

    lookup = create_lookup(lstgs=get_lstgs(part))
    topickle(lookup, PARTS_DIR + '{}/{}.pkl'.format(part, LOOKUP))


if __name__ == "__main__":
    main()

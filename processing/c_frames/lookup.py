import pandas as pd
from constants import PARTS_DIR, DAY
from featnames import LOOKUP, META, START_TIME, END_TIME, SLR_BO_CT, \
    START_PRICE, DEC_PRICE, ACC_PRICE, START_DATE
from processing.util import get_lstgs, feat_to_pctile
from utils import topickle, input_partition, load_feats


def create_lookup(lstgs=None):
    # load data
    listings = load_feats('listings', lstgs=lstgs)

    # start time instead of start date
    start_time = listings[START_DATE].astype('int64') * DAY
    start_time = start_time.rename(START_TIME)

    # subset features
    lookup = listings[[META, START_PRICE, DEC_PRICE, ACC_PRICE]]
    lookup = pd.concat([lookup, start_time, listings[END_TIME]], axis=1)

    # add count of past best offer listings for seller
    lookup[SLR_BO_CT] = feat_to_pctile(listings[SLR_BO_CT],
                                       reverse=True,
                                       feat=SLR_BO_CT)

    return lookup


def main():
    part = input_partition()
    print('{}/{}'.format(part, LOOKUP))

    lookup = create_lookup(lstgs=get_lstgs(part))
    topickle(lookup, PARTS_DIR + '{}/{}.pkl'.format(part, LOOKUP))


if __name__ == "__main__":
    main()

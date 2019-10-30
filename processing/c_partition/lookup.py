import sys, random
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *


if __name__ == "__main__":
    # listings
    lstgs = load(CLEAN_DIR + 'listings.gz')
    
    # partition by seller
    partitions = load(PARTS_DIR + 'partitions.gz')

    # save lookup file
    lstgs = lstgs[['meta', 'start_date', 'end_time', \
        'start_price', 'decline_price', 'accept_price']]
    for part, idx in partitions.items():
        lookup = lstgs.reindex(index=idx)
        dump(lookup, PARTS_DIR + '%s/lookup.gz' % part)
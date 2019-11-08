import sys
from compress_pickle import dump, load
import pandas as pd, numpy as np
from constants import *

LVARS = ['cat', 'cndtn', 'start_date', 'end_time', \
    'start_price', 'start_price_pctile', 'arrival_rate', 'flag']
TVARS = ['start_time', 'byr_hist', 'bin']
OVARS = ['clock', 'price', 'accept', 'reject', 'censored']


if __name__ == '__main__':
	# parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num

    # read in data frames
    L = load(CLEAN_DIR + 'listings.pkl')
    T = load(CLEAN_DIR + 'threads.pkl')
    O = load(CLEAN_DIR + 'offers.pkl')

    # slr features
    if num == 1:
    	chunk('slr', L, T, O)

    # cat features
    elif num == 2:
    	L = L[LVARS]
    	T = T[TVARS]
    	O = O[OVARS]
    	chunk('cat', L, T, O)
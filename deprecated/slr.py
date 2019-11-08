import sys
from compress_pickle import dump, load
import pandas as pd, numpy as np
from constants import *


# read in data frames
L = load(CLEAN_DIR + 'listings.gz')
T = load(CLEAN_DIR + 'threads.gz')
O = load(CLEAN_DIR + 'offers.gz')

chunk('slr', L, T, O)
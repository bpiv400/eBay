from pandas.tseries.holiday import USFederalHolidayCalendar as Calendar
import torch
import numpy as np
from platform import platform

# strings for referencing quantities related to buyer and seller interface
SLR_PREFIX = 'slr'
BYR_PREFIX = 'byr'
ARRIVAL_PREFIX = 'arrival'

# vocabulary size for embeddings
VOCAB_SIZE = 32

# create new chunk once number of listings exceeds CUTOFF
CUTOFF = 1e5

# for partitioning
SEED = 123456
SHARES = {'train_models': 1/3, 'train_rl': 1/3}

# count concessions within this range as 1/2
TOL_HALF = 0.02

# optimization parameters
EPOCHS = 100
NUM_WORKERS = 4 if torch.cuda.is_available() else 0

# paths and directories
if torch.cuda.is_available() and 'Windows' not in platform():
	PREFIX = '/data/eBay'
else:
	PREFIX = 'data'

CLEAN_DIR = '%s/clean/' % PREFIX
CHUNKS_DIR = '%s/chunks/' % PREFIX
FEATS_DIR = '%s/feats/' % PREFIX
PARTS_DIR = '%s/partitions/' % PREFIX
REWARDS_DIR = '%s/rewards/' % PREFIX
PCTILE_DIR = '%s/pctile/' % PREFIX
W2V_DIR = '%s/w2v/' % PREFIX
INPUT_DIR = '%s/inputs/' % PREFIX
SUMMARY_DIR = 'outputs/summary/'
MODEL_DIR = 'outputs/models/'

# partitions
PARTITIONS = ['train_models', 'train_rl', 'test']

# minimum number of listings that define a category
MIN_COUNT = 1000

# delete activity after lstg is open MAX_DAYS
MAX_DAYS = 31

# maximal delay times
MAX_DELAY = {
	'arrival': MAX_DAYS * 24 * 3600,
	SLR_PREFIX: 2 * 24 * 3600,
	BYR_PREFIX: 14 * 24 * 3600
}

# intervals for checking offer arrivals
INTERVAL = {
	'arrival': 4 * 60 * 60,
	SLR_PREFIX: 15 * 60,
	BYR_PREFIX: 90 * 60
}

INTERVAL_COUNTS = {
	'arrival': int(MAX_DELAY['arrival'] / INTERVAL['arrival']),
	SLR_PREFIX: int(MAX_DELAY[SLR_PREFIX] / INTERVAL[SLR_PREFIX]),
	BYR_PREFIX: int(MAX_DELAY[BYR_PREFIX] / INTERVAL[BYR_PREFIX]),
	'{}_7'.format(BYR_PREFIX): \
		int(MAX_DELAY[SLR_PREFIX] / INTERVAL[BYR_PREFIX])
}

# quantiles of byr_hist distribution
HIST_QUANTILES = 10

# model names
MODELS = ['arrival','hist', 'delay_byr', 'delay_slr', 'con_byr', 'con_slr', \
	'msg_byr', 'msg_slr']

# number of observations in small dataset
N_SMALL = 100000

# organizing hierarchy
LEVELS = ['slr', 'cat', 'cndtn', 'lstg']

# for lstg feature construction
ASIS_FEATS = ['store', 'slr_us', 'fast', 'slr_bos', 'slr_lstgs', \
	'fdbk_score', 'fdbk_pstv', 'start_price_pctile']

# indices for byr and slr offers
IDX = {
	BYR_PREFIX: [1, 3, 5, 7],
	SLR_PREFIX: [2, 4, 6]
}

# date range and holidays
START = '2012-06-01 00:00:00'
END = '2013-05-31 23:59:59'
HOLIDAYS = Calendar().holidays(start=START, end=END)

# quantiles of accept_norm distribution
QUANTILES = [0.25, 0.5, 0.75, 1]


EPOCH_TIME = 'epoch_time'

ARRIVAL_PERIODS = 31 * 24


# data types for csv read
OTYPES = {'lstg': 'int64',
		  'thread': 'int64',
		  'index': 'uint8',
		  'clock': 'int64', 
		  'price': 'float64', 
		  'accept': bool,
		  'reject': bool,
		  'censored': bool,
		  'message': bool}

TTYPES = {'lstg': 'int64',
		  'thread': 'int64',
		  'byr': 'int64',
		  'byr_hist': 'int64',
		  'bin': bool,
		  'byr_us': bool}

LTYPES = {'lstg': 'int64',
		  'slr': 'int64',
		  'meta': 'uint8',
		  'cat': str,
		  'cndtn': 'uint8',
		  'start_date': 'uint16',
		  'end_time': 'int64',
		  'fdbk_score': 'int64',
		  'fdbk_pstv': 'int64',
		  'start_price': 'float64',
		  'photos': 'uint8',
		  'slr_lstgs': 'int64',
		  'slr_bos': 'int64',
		  'decline_price': 'float64',
		  'accept_price': 'float64',
		  'store': bool,
		  'slr_us': bool,
		  'flag': bool,
		  'fast': bool}
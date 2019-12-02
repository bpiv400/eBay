from pandas.tseries.holiday import USFederalHolidayCalendar as Calendar
import torch
import numpy as np
from platform import platform
import multiprocessing as mp

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
EPOCHS = 10
NUM_WORKERS = mp.cpu_count() if torch.cuda.is_available() else 0
MBSIZE_VALIDATION = 2000

# paths and directories
if 'Ubuntu' in platform():
	PREFIX = '/data/eBay'
elif 'centos' in platform():
	PREFIX = '/home/opim/etangr/weka/eBay'
elif 'Windows' in platform():
	PREFIX = 'data'
else:
	PREFIX = '/home/opim/etangr/weka/eBay'

CLEAN_DIR = '%s/clean/' % PREFIX
CHUNKS_DIR = '%s/chunks/' % PREFIX
FEATS_DIR = '%s/feats/' % PREFIX
PARTS_DIR = '%s/partitions/' % PREFIX
REWARDS_DIR = '%s/simulator/' % PREFIX
PCTILE_DIR = '%s/pctile/' % PREFIX
W2V_DIR = '%s/w2v/' % PREFIX
INPUT_DIR = '%s/inputs/' % PREFIX
SUMMARY_DIR = 'outputs/summary/'
MODEL_DIR = 'outputs/models/'
LOG_DIR = 'outputs/logs/'

# partitions
PARTITIONS = ['train_models', 'train_rl', 'test']

# minimum number of listings that define a category
MIN_COUNT = 1000

# delete activity after lstg is open MAX_DAYS
MAX_DAYS = 31
ARRIVAL_PERIODS = MAX_DAYS * 24

# maximal delay times
MAX_DELAY = {
	ARRIVAL_PREFIX: MAX_DAYS * 24 * 3600,
	SLR_PREFIX: 2 * 24 * 3600,
	BYR_PREFIX: 14 * 24 * 3600
}

# intervals for checking offer arrivals
INTERVAL = {
	ARRIVAL_PREFIX: 4 * 60 * 60,
	SLR_PREFIX: 15 * 60,
	BYR_PREFIX: 90 * 60
}

INTERVAL_COUNTS = {
	ARRIVAL_PREFIX: int(MAX_DELAY[ARRIVAL_PREFIX] / INTERVAL[ARRIVAL_PREFIX]),
	SLR_PREFIX: int(MAX_DELAY[SLR_PREFIX] / INTERVAL[SLR_PREFIX]),
	BYR_PREFIX: int(MAX_DELAY[BYR_PREFIX] / INTERVAL[BYR_PREFIX]),
	BYR_PREFIX + '_7': int(MAX_DELAY[SLR_PREFIX] / INTERVAL[BYR_PREFIX])
}

# quantiles of byr_hist distribution
HIST_QUANTILES = 10

# model names
MODELS = ['arrival', 'hist', 'delay_byr', 'delay_slr', 'con_byr', 'con_slr', \
	'msg_byr', 'msg_slr']

# number of observations in small dataset
N_SMALL = 100000

# organizing hierarchy
LEVELS = ['slr', 'cat', 'cndtn', 'lstg']

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

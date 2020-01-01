import os
from pandas.tseries.holiday import USFederalHolidayCalendar as Calendar
from platform import platform

# strings for referencing quantities related to buyer and seller interface
SLR_PREFIX = 'slr'
BYR_PREFIX = 'byr'
ARRIVAL_PREFIX = 'arrival'

# count concessions within this range as 1/2
TOL_HALF = 0.02

# paths and directories
if 'Ubuntu' in platform():		# Etan's box
	PREFIX = '/data/eBay'
elif 'Windows' in platform():	# Barry's laptop
	PREFIX = 'A:/ebay/data'
else:							# cluster and AWS
	PREFIX = os.path.expanduser('~/weka/eBay')

PARTS_DIR = '%s/partitions/' % PREFIX
ENV_SIM_DIR = '%s/envSimulator/' % PREFIX
OUTPUT_DIR = '%s/outputs/' % PREFIX
INPUT_DIR = '%s/inputs/' % PREFIX
FEATNAMES_DIR = '%sfeatnames/' % INPUT_DIR
MODEL_DIR = '%smodels/' % OUTPUT_DIR
REINFORCE_DIR = '%s/reinforce' % PREFIX
REINFORCE_INPUT_DIR = '%s/input' % REINFORCE_DIR

# partitions
PARTITIONS = ['train_models', 'train_rl', 'test_rl', 'test']

# delete activity after lstg is open MAX_DAYS
MAX_DAYS = 31

# temporal constants
HOUR = 3600
DAY = 24 * HOUR
MONTH = MAX_DAYS * DAY
EXPIRATION = 2 * DAY

# maximal delay times
MAX_DELAY = {
	ARRIVAL_PREFIX: MAX_DAYS * 24 * 3600,
	SLR_PREFIX: 2 * 24 * 3600,
	BYR_PREFIX: 14 * 24 * 3600
}

# intervals for checking offer arrivals
INTERVAL = {
	ARRIVAL_PREFIX: 60 * 60,
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
MODELS = ['arrival', 'hist', \
		  'delay_byr', 'delay_slr', \
		  'con_byr', 'con_slr', \
		  'msg_byr', 'msg_slr']

# number of chunks for environment simulation
SIM_CHUNKS = 1000

# indices for byr and slr offers
IDX = {
	BYR_PREFIX: [1, 3, 5, 7],
	SLR_PREFIX: [2, 4, 6]
}

# date range and holidays
START = '2012-06-01 00:00:00'
END = '2013-05-31 23:59:59'
HOLIDAYS = Calendar().holidays(start=START, end=END)

# groups for embedding layers
EMBEDDING_GROUPS = {'w2v': ['lstg', 'w2v_slr', 'w2v_byr'],
					'other': ['lstg', 'cat', 'cndtn', 'slr']}
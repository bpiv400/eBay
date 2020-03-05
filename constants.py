import os
from pandas.tseries.holiday import USFederalHolidayCalendar as Calendar
from platform import platform
from featnames import DELAY, CON, MSG

# strings for referencing quantities related to buyer and seller interface
SLR_PREFIX = 'slr'
BYR_PREFIX = 'byr'
ARRIVAL_PREFIX = 'arrival'

# count concessions within this range as 1/2
TOL_HALF = 0.02

# paths and directories
if 'Ubuntu' in platform():  # Etan's box
    PREFIX = '/data/eBay'
elif 'debian' in platform():
    PREFIX = os.path.expanduser('~/shared/ebay')
elif 'Windows' in platform() and 'A:' in os.getcwd():  # Barry's pc
    PREFIX = 'A:/ebay'
elif 'Windows' in platform() and 'C:' in os.getcwd():  # Barry's laptop
    PREFIX = os.path.expanduser('~/ebay')
else:  # cluster and AWS
    PREFIX = os.path.expanduser('~/weka/eBay')

PARTS_DIR = '%s/partitions/' % PREFIX
ENV_SIM_DIR = '%s/envSimulator/' % PREFIX
OUTPUT_DIR = '%s/outputs/' % PREFIX
INPUT_DIR = '%s/inputs/' % PREFIX
INDEX_DIR = '%s/index/' % PREFIX
FEATNAMES_DIR = '%sfeatnames/' % INPUT_DIR
MODEL_DIR = '%smodels/' % OUTPUT_DIR
REINFORCE_DIR = '%s/agent/' % PREFIX

PARAMS_PATH = INPUT_DIR + 'params.pkl'

# partitions
TRAIN_MODELS = 'train_models'
TRAIN_RL = 'train_rl'
VALIDATION = 'test_rl'
TEST = 'test'
PARTITIONS = [TRAIN_MODELS, TRAIN_RL, VALIDATION, TEST]
SMALL = 'small'

# delete activity after lstg is open MAX_DAYS
MAX_DAYS = 31

# temporal constants
MINUTE = 60
HOUR = 60 * MINUTE
DAY = 24 * HOUR
MONTH = MAX_DAYS * DAY
EXPIRATION = 2 * DAY

# maximal delay times
MAX_DELAY = {1: MONTH,
             2: 2 * DAY,
             3: 14 * DAY,
             4: 2 * DAY,
             5: 14 * DAY,
             6: 2 * DAY,
             7: 2 * DAY}

# concessions that denote an (almost) even split
SPLIT_PCTS = [.49, .50, .51]

# quantiles of byr_hist distribution
HIST_QUANTILES = 10

# multiplier for concession
CON_MULTIPLIER = 100

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

# model names
FIRST_ARRIVAL_MODEL = 'first_arrival'
INTERARRIVAL_MODEL = 'next_arrival'
BYR_HIST_MODEL = 'hist'

# model sets
ARRIVAL_MODELS = [FIRST_ARRIVAL_MODEL, INTERARRIVAL_MODEL, BYR_HIST_MODEL]
DELAY_MODELS = ['{}{}'.format(DELAY, i) for i in range(2, 8)]
CON_MODELS = ['{}{}'.format(CON, i) for i in range(1, 8)]
MSG_MODELS = ['{}{}'.format(MSG, i) for i in range(1, 7)]
OFFER_MODELS = DELAY_MODELS + CON_MODELS + MSG_MODELS
MODELS = ARRIVAL_MODELS + OFFER_MODELS

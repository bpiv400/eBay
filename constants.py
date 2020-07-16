import os
from pandas.tseries.holiday import USFederalHolidayCalendar as Calendar
from platform import platform
from featnames import DELAY, CON, MSG

# strings for referencing quantities related to buyer and seller interface
SLR = 'slr'
BYR = 'byr'
ARRIVAL = 'arrival'
DROPOUT = 'dropout'

# count concessions within this range as 1/2
TOL_HALF = 0.02

# paths and directories
if 'Ubuntu' in platform():  # Etan's new box
    PREFIX = os.path.expanduser('~/eBay/data')
elif 'debian' in platform():
    PREFIX = os.path.expanduser('~/shared/ebay/data')
elif 'Windows' in platform() and 'A:' in os.getcwd():  # Barry's pc
    PREFIX = 'A:/ebay/data'
elif 'Windows' in platform() and 'C:' in os.getcwd():  # Barry's laptop
    PREFIX = os.path.expanduser('~/ebay/data')
elif 'Darwin' in platform():  # Etan's Mac laptop
    PREFIX = os.path.expanduser('~/eBay/data')
else:  # cluster and AWS
    PREFIX = os.path.expanduser('~/weka/eBay')

MODEL_PARTS_DIR = '{}/partitions/models/'.format(PREFIX)
AGENT_PARTS_DIR = '{}/partitions/agent/'.format(PREFIX)
INDEX_DIR = '{}/index/'.format(PREFIX)
PCTILE_DIR = '{}/pctile/'.format(PREFIX)
CLEAN_DIR = '{}/clean/'.format(PREFIX)
W2V_DIR = '{}/w2v/'.format(PREFIX)
FEATS_DIR = '{}/feats/'.format(PREFIX)

INPUT_DIR = '{}/inputs/'.format(PREFIX)
SIZES_DIR = INPUT_DIR + 'sizes/'
FEATNAMES_DIR = INPUT_DIR + 'featnames/'

OUTPUT_DIR = '{}/outputs/'.format(PREFIX)
LOG_DIR = OUTPUT_DIR + 'logs/'
MODEL_DIR = OUTPUT_DIR + 'models/'
PLOT_DIR = OUTPUT_DIR + 'plots/'
AGENT_DIR = OUTPUT_DIR + 'agent/'

# partitions
TRAIN_MODELS = 'sim'
TRAIN_RL = 'rl'
VALIDATION = 'valid'
TEST = VALIDATION  # TODO: rename to 'testing' when using real testing data
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
MAX_DELAY_ARRIVAL = MONTH
MAX_DELAY_TURN = 2 * DAY

# concessions that denote an (almost) even split
SPLIT_PCTS = [.49, .50, .51]

# quantiles of byr_hist distribution
HIST_QUANTILES = 10

# multiplier for concession
CON_MULTIPLIER = 100

# indices for byr and slr offers
IDX = {
    BYR: [1, 3, 5, 7],
    SLR: [2, 4, 6]
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

# censored models
CENSORED_MODELS = [FIRST_ARRIVAL_MODEL, INTERARRIVAL_MODEL] + DELAY_MODELS

# discriminator models
DISCRIM_LISTINGS = 'listings'
DISCRIM_THREADS = 'threads'
DISCRIM_THREADS_NO_TF = 'threads_no_tf'
DISCRIM_MODELS = [DISCRIM_LISTINGS, DISCRIM_THREADS, DISCRIM_THREADS_NO_TF]

# policy initializations
AGENTS = [SLR, BYR]
POLICY_SLR = 'policy_slr'
POLICY_BYR = 'policy_byr'
POLICY_MODELS = [POLICY_SLR, POLICY_BYR]

TURN_FEATS = {
    BYR: ['t1', 't3', 't5'],
    SLR: ['t2', 't4']
}

# outcome types
SIM = 'simulation'
OBS = 'data'

# normalization type
MODEL_NORM = 'batch'

# fee constants
LISTING_FEE = .03

# meta categories with sale fees != .09 * price
META_7 = [11116, 619]
META_6 = [58058, 1249, 625, 293, 15032]

# threshold for likelihood of no arrivals
NO_ARRIVAL_CUTOFF = .50 ** (1.0 / 12)

# fixed random seed
SEED = 123456

# number of chunks
NUM_CHUNKS = 60

# multiprocessing and RL training
NUM_RL_WORKERS = 16

import os
from pandas.tseries.holiday import USFederalHolidayCalendar as Calendar
from platform import platform
from featnames import SLR, BYR, TRAIN_MODELS, TRAIN_RL, VALIDATION

# paths and directories
if 'Ubuntu' in platform():  # Etan's box(es)
    PREFIX = os.path.expanduser('/data/eBay')
elif 'debian' in platform():
    PREFIX = os.path.expanduser('~/shared/ebay/data')
elif 'Windows' in platform() and 'A:' in os.getcwd():  # Barry's pc
    PREFIX = 'A:/ebay/data'
elif 'Windows' in platform() and 'C:' in os.getcwd():  # Barry's laptop
    PREFIX = os.path.expanduser('~/ebay/data')
else:  # cluster and AWS
    PREFIX = os.path.expanduser('~/eBay/data')

PARTS_DIR = '{}/partitions/'.format(PREFIX)     # post-partition features
SIM_DIR = '{}/sim/'.format(PREFIX)              # simulated threads and offers
INDEX_DIR = '{}/index/'.format(PREFIX)          # indices for input files
PCTILE_DIR = '{}/pctile/'.format(PREFIX)        # percentiles of features
CLEAN_DIR = '{}/clean/'.format(PREFIX)          # cleaned csvs
W2V_DIR = '{}/w2v/'.format(PREFIX)              # for word2vec features
FEATS_DIR = '{}/feats/'.format(PREFIX)          # pre-partion features

INPUT_DIR = '{}/inputs/'.format(PREFIX)         # inputs for models
SIZES_DIR = INPUT_DIR + 'sizes/'                # for initializing models
FEATNAMES_DIR = INPUT_DIR + 'featnames/'        # for testing

OUTPUT_DIR = '{}/outputs/'.format(PREFIX)       # for saving outputs
LOG_DIR = OUTPUT_DIR + 'logs/'                  # model logs
MODEL_DIR = OUTPUT_DIR + 'models/'              # trained models
PLOT_DIR = OUTPUT_DIR + 'plots/'                # for creating figures
AGENT_DIR = OUTPUT_DIR + 'agent/'               # agents logs and models

DATE_FEATS_PATH = FEATS_DIR + 'date_feats.pkl'
META_PATH = CLEAN_DIR + 'meta.csv'

FIG_DIR = os.path.expanduser('~/Dropbox/eBay/figures/')  # for saving figures

# for splitting data
SHARES = {TRAIN_MODELS: 0.75, TRAIN_RL: 0.1, VALIDATION: 0.05}

# listing window stays open this many days
MAX_DAYS = 8

# temporal constants
MINUTE = 60
HOUR = 60 * MINUTE
DAY = 24 * HOUR

# maximal delay times
MAX_DELAY_ARRIVAL = MAX_DAYS * DAY
MAX_DELAY_TURN = 2 * DAY

# intervals
INTERVAL_TURN = int(5 * MINUTE)
INTERVAL_ARRIVAL = int(5 * MINUTE)
INTERVAL_CT_TURN = int(MAX_DELAY_TURN / INTERVAL_TURN)
INTERVAL_CT_ARRIVAL = int(MAX_DELAY_ARRIVAL / INTERVAL_ARRIVAL)

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

# normalization type
MODEL_NORM = 'batch'

# meta categories for collectibles
COLLECTIBLES = [1, 237, 260, 550, 870, 11116, 20081, 45100, 64482]

# fixed random seed
SEED = 123456

# number of chunks
NUM_CHUNKS = 1024

# simulation counts
OUTCOME_SIMS = 10
VALUE_SIMS = 100
ARRIVAL_SIMS = 10

# features to drop from 'lstg' grouping for byr agent
BYR_DROP = ['lstg_ct', 'bo_ct',
            'auto_decline', 'has_decline',
            'auto_accept', 'has_accept']

# for precision issues
EPS = 1e-8

# dropout options
DROPOUT_GRID = []
for j in range(8):
    for i in range(j+1):
        if j - i <= 1:
            DROPOUT_GRID.append((float(i) / 10, float(j) / 10))

# number of concessions available to agent
NUM_COMMON_CONS = 6

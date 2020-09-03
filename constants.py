import os
from pandas.tseries.holiday import USFederalHolidayCalendar as Calendar
from platform import platform
from featnames import DELAY, CON, MSG, SLR, BYR

# paths and directories
if 'Ubuntu' in platform():  # Etan's box(es)
    PREFIX = os.path.expanduser('/data/eBay')
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

PARTS_DIR = '{}/partitions/'.format(PREFIX)     # post-partition features
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

FIG_DIR = os.path.expanduser('~/eBay/figures/')  # for saving figures

# partitions
TRAIN_MODELS = 'sim'
TRAIN_DISCRIM = 'discrim'
RL_SLR = 'rl_slr'
RL_BYR = 'rl_byr'
VALIDATION = 'valid'
TEST = VALIDATION  # TODO: rename to 'testing' when using real testing data
PARTITIONS = [TRAIN_MODELS, TRAIN_DISCRIM, RL_SLR, RL_BYR, VALIDATION, TEST]
SIM_PARTITIONS = [TRAIN_MODELS, VALIDATION, TEST]
AGENT_PARTITIONS = [RL_SLR, RL_BYR, VALIDATION, TEST]

# for splitting data
SHARES = {TRAIN_MODELS: 0.75, RL_SLR: 0.05, RL_BYR: 0.05, VALIDATION: 0.05}

# listing window stays open this many days
MAX_DAYS = 8

# temporal constants
MINUTE = 60
HOUR = 60 * MINUTE
DAY = 24 * HOUR
EXPIRATION = 2 * DAY

# maximal delay times
MAX_DELAY_ARRIVAL = MAX_DAYS * DAY
MAX_DELAY_TURN = 2 * DAY

# intervals
INTERVAL_TURN = int(5 * MINUTE)
INTERVAL_ARRIVAL = int(15 * MINUTE)
INTERVAL_CT_TURN = int(MAX_DELAY_TURN / INTERVAL_TURN)
INTERVAL_CT_ARRIVAL = int(MAX_DELAY_ARRIVAL / INTERVAL_ARRIVAL)

# concessions that denote an (almost) even split
SPLIT_PCTS = [.49, .50, .51]

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
CENSORED_MODELS = [INTERARRIVAL_MODEL] + DELAY_MODELS

# discriminator models
DISCRIM_MODEL = 'discrim'

# policy initializations
POLICY_SLR = 'policy_slr'
POLICY_BYR = 'policy_byr'
POLICY_MODELS = [POLICY_SLR, POLICY_BYR]

# normalization type
MODEL_NORM = 'batch'

# fee constants
LISTING_FEE = .03

# meta categories with sale fees != .09 * price
META_7 = [11116, 619]
META_6 = [58058, 1249, 625, 293, 15032]

# meta categories for collectibles
COLLECTIBLES = [1, 237, 260, 550, 870, 11116, 20081, 45100, 64482]

# fixed random seed
SEED = 123456

# number of chunks
NUM_CHUNKS = 64

# features to drop from 'lstg' grouping for byr agent
BYR_DROP = ['lstg_ct', 'bo_ct', 'auto_decline', 'has_decline']

# for precision issues
EPS = 1e-8

# dropout options
DROPOUT_GRID = []
for j in range(8):
    for i in range(j+1):
        if j - i <= 1:
            DROPOUT_GRID.append((float(i) / 10, float(j) / 10))

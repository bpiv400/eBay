from constants import INPUT_DIR, PARTS_DIR
import torch

INTERACT = False
VERBOSE = True

# time feats
SLR_OFFERS = 'slr_offers'
BYR_OFFERS = 'byr_offers'
SLR_OFFERS_OPEN = 'slr_offers_open'
BYR_OFFERS_OPEN = 'byr_offers_open'
SLR_BEST = 'slr_best'
BYR_BEST = 'byr_best'
SLR_BEST_OPEN = 'slr_best_open'
BYR_BEST_OPEN = 'byr_best_open'
THREAD_COUNT = 'thread_count'
SLR_OFFERS_RECENT = 'slr_offers_recent'
BYR_OFFERS_RECENT = 'byr_offers_recent'
SLR_BEST_RECENT = 'slr_best_recent'
BYR_BEST_RECENT = 'byr_best_recent'

TIME_FEATS = [
    SLR_OFFERS,
    SLR_BEST,
    SLR_OFFERS_OPEN,
    SLR_BEST_OPEN,
    SLR_OFFERS_RECENT,
    SLR_BEST_RECENT,
    BYR_OFFERS,
    BYR_BEST,
    BYR_OFFERS_OPEN,
    BYR_BEST_OPEN,
    BYR_OFFERS_RECENT,
    BYR_BEST_RECENT,
    THREAD_COUNT
]

# outcomes
BYR_OUTCOMES = ['days', 'delay', 'con', 'norm', 'split', 'msg']
SLR_OUTCOMES = BYR_OUTCOMES + ['reject', 'auto', 'exp']
REJ_POS = SLR_OUTCOMES.index('reject')
AUTO_POS = SLR_OUTCOMES.index('auto')
EXPIRE_POS = SLR_OUTCOMES.index('exp')
NORM_POS = SLR_OUTCOMES.index('norm')
CON_POS = SLR_OUTCOMES.index('con')
MSG_POS = SLR_OUTCOMES.index('msg')
DAYS_POS = SLR_OUTCOMES.index('days')
DELAY_POS = SLR_OUTCOMES.index('delay')
SPLIT_POS = SLR_OUTCOMES.index('split')

# turn indices
SLR_TURN_INDS = ['t1', 't2']
BYR_TURN_INDS = SLR_TURN_INDS + ['t3']

# clock feats
CLOCK_FEATS = ['holiday', 'dow0', 'dow1', 'dow2', 'dow3', 'dow4', 'dow5', 'minute_of_day']
MONTHS_SINCE_LSTG = 'months_since_lstg'
DAYS_SINCE_THREAD = 'days_since_thread'
DURATION = 'duration'
INT_REMAINING = 'remaining'

# hist feature
BYR_HIST = 'byr_hist'

# dataset dictionary keys
X_LSTG = 'x_lstg'
LOOKUP = 'lookup'

# filenames
PARTITION = 'train_rl'
DATA_DIR = '{}{}/'.format(PARTS_DIR, PARTITION)
COMPOSER_DIR = '{}composer/'.format(INPUT_DIR)
X_LSTG_COLS_PATH = '{}{}.pkl'.format(COMPOSER_DIR, X_LSTG)
REWARD_EXPERIMENT_PATH = 'repo/rlenv/rewards/experiments.csv'
LOOKUP_FILENAME = 'lookup.gz'
PARAMS_PATH = '{}params.csv'.format(INPUT_DIR)

# temporal constants
MONTH = 31 * 24 * 3600
DAY = 24 * 3600
HOUR = 3600
EXPIRATION = 48 * 60 * 60


# zero vectors
# TODO: check whether there are similar outcome vectors
ZERO_SLR_OUTCOMES = torch.zeros(len(SLR_OUTCOMES))
ZERO_SLR_OUTCOMES[[7, 8]] = 1

CLOCK_ZEROS = torch.zeros(len(CLOCK_FEATS))

FIXED = 'fixed'
TIME = 'time'

# parameter names
SIM_COUNT = 'n'

# fee constants
ANCHOR_STORE_INSERT = .03

# reject outcomes
AUTO_REJ_OUTCOMES = torch.zeros(len(SLR_OUTCOMES))
AUTO_REJ_OUTCOMES[[7, 8]] = 1  # automatic, rejected
EXP_REJ_OUTCOMES = torch.zeros(len(SLR_OUTCOMES))
EXP_REJ_OUTCOMES[[0, 8]] = 1  # full delay, rejected, expired

# lookup column names
LSTG = 'lstg'
SLR = 'slr'
STORE = 'store'
META = 'meta'
START_DAY = 'start_date'
START_PRICE = 'start_price'
DEC_PRICE = 'decline_price'
ACC_PRICE = 'accept_price'

# meta categories with sale fees != .09 * price
META_7 = [21, 10]
META_6 = [32, 14, 11, 7, 28]

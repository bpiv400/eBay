from constants import INPUT_DIR, PARTS_DIR
import torch

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
BYR_OUTCOMES = ['delay', 'days', 'con', 'norm', 'split', 'msg']
SLR_OUTCOMES = BYR_OUTCOMES + ['reject', 'auto', 'exp']


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
X_LSTG_FILENAME = '{}{}.hdf5'.format(DATA_DIR, X_LSTG)
COMPOSER_DIR = '{}composer/'.format(INPUT_DIR)
REWARD_EXPERIMENT_PATH = 'repo/rlenv/rewards/experiments.csv'
LOOKUP_FILENAME = 'lookup.gz'

# temporal constants
MONTH = 31 * 24 * 3600
DAY = 24 * 3600
HOUR = 3600
EXPIRATION = 48 * 60 * 60

# base map names
LSTG_MAP = 'lstg'
BYR_HIST_MAP = 'byr_hist'
# curr turn maps
OUTCOMES_MAP = 'outcomes'
CLOCK_MAP = 'clock'
TIME_MAP = 'time'
TURN_IND_MAP = 'turn_inds'
DIFFS_MAP = 'diffs'
# other turn maps
O_CLOCK_MAP = 'other_clock'
O_OUTCOMES_MAP = 'other_outcomes'
O_TIME_MAP = 'other_time'
O_DIFFS_MAP = 'other_diffs'
# last turn maps
L_CLOCK_MAP = 'last_clock'
L_OUTCOMES_MAP = 'last_outcomes'
L_TIME_MAP = 'last_time'
# misc time maps
INT_REMAIN_MAP = 'intervals'
DUR_MAP = 'duration'
MONTHS_LSTG_MAP = 'months_lstg'
DAYS_THREAD_MAP = 'days_thread'
# SIZE map
SIZE = 'size'

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

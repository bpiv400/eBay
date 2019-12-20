"""
Constants used by rlenv scripts
"""
from constants import INPUT_DIR

INTERACT = False
VERBOSE = False  # verbose has higher priority than silent
SILENT = False  # output nothing

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

# list of time feats -- gives the order output by TimeFeatures.get_feats(...)
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

# clock feats
CLOCK_FEATS = ['holiday', 'dow0', 'dow1', 'dow2', 'dow3', 'dow4', 'dow5', 'minute_of_day']

# turn feats
TURN_FEATS = ['t1', 't2', 't3']

# outcomes
DAYS = 'days'
DELAY = 'delay'
CON = 'con'
NORM = 'norm'
SPLIT = 'split'
MSG = 'msg'
REJECT = 'reject'
AUTO = 'auto'
EXP = 'exp'

# list of buyer outcomes
BYR_OUTCOMES = [
    DAYS,
    DELAY,
    CON,
    NORM,
    SPLIT,
    MSG,
]
# list of seller outcomes
SLR_OUTCOMES = BYR_OUTCOMES + [REJECT, AUTO, EXP]

# dictionaries mapping turn numbers to lists of feats in those turns
ALL_CLOCK_FEATS = dict()
ALL_TIME_FEATS = dict()
ALL_OUTCOMES = dict()
for i in range(7):
    ALL_CLOCK_FEATS[i + 1] = ['{}_{}'.format(feat, i + 1) for feat in CLOCK_FEATS]
    ALL_TIME_FEATS[i + 1] = ['{}_{}'.format(feat, i + 1) for feat in TIME_FEATS]
    if (i + 1) % 2 == 1:
        ALL_OUTCOMES[i + 1] = ['{}_{}'.format(feat, i + 1) for feat in BYR_OUTCOMES]
    else:
        ALL_OUTCOMES[i + 1] = ['{}_{}'.format(feat, i + 1) for feat in SLR_OUTCOMES]

# outcomes associated with recurrent models
MONTHS_SINCE_LSTG = 'months_since_lstg'
DURATION = 'duration'
INT_REMAINING = 'remaining'

# hist feature
BYR_HIST = 'byr_hist'

# dataset dictionary keys
X_LSTG = 'x_lstg'
LOOKUP = 'lookup'

# filenames
COMPOSER_DIR = '{}composer/'.format(INPUT_DIR)  # location of composer
AGENT_FEATS_FILENAME = 'agent_feats.xlsx'  # location of file containing lists of features for all agents
LOOKUP_FILENAME = 'lookup.gz'
X_LSTG_FILENAME = 'x_lstg.gz'
# partition subdir names
SIM_CHUNKS_DIR = 'chunks'
SIM_VALS_DIR = 'vals'
SIM_DISCRIM_DIR = 'discrim'

# temporal constants
MONTH = 31 * 24 * 3600
DAY = 24 * 3600
HOUR = 3600
EXPIRATION = 48 * 60 * 60

# fee constants
ANCHOR_STORE_INSERT = .03

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

# environment simulator param names
NUM_CHUNKS = 512
VAL_SE_CHECK = 50  # how many trials should pass between computations of standard error
MIN_SALES = 20  # minimum number of sales before value estimate can be considered stable
# list of values that will be used as standard error maximums
SE_TOLS = [.5, .75, 1.0, 1.25]  # SE[i] is max for trail i * SE_RELAX_WIDTH,... (i + 1) * RELAX_WIDTH - 1
SE_RELAX_WIDTH = 10000  # number of trials that passes between standard error maximum relaxations
SIM_COUNT = 100  # number of times each lstg should be simulated when generating discrim inputs
MAX_RECORDER_SIZE = 2e9 # maximum size of recorder object in bytes before dumping output

# space names
ACTION_SPACE_NAME = 'NegotiationActionSpace'
OBS_SPACE_NAME = 'NegotiationObsSpace'
SELLER_HORIZON = 100

# agent names (see AGENTS_FEATS_FILENAME)
INFO_AGENTS = ['byr', 'slr0', 'slr1']

# outcome tuple names
SALE = 'sale'
DUR = 'dur'
PRICE = 'price'

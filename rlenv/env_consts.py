from constants import INPUT_DIR, MAX_DAYS

INTERACT = False

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

# composer maps
SIZE = 'size'
LSTG_MAP = 'lstg'
THREAD_MAP = 'thread'
TURN_IND_MAP = 'turns'
X_TIME_MAP = 'x_time'

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

BYR_OUTCOMES = [
    DAYS,
    DELAY,
    CON,
    NORM,
    SPLIT,
    MSG,
]
SLR_OUTCOMES = BYR_OUTCOMES + [REJECT, AUTO, EXP]

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



# turn indices
SLR_TURN_INDS = ['t1', 't2']
BYR_TURN_INDS = SLR_TURN_INDS + ['t3']


MONTHS_SINCE_LSTG = 'months_since_lstg'
DURATION = 'duration'
INT_REMAINING = 'remaining'

# hist feature
BYR_HIST = 'byr_hist'

# dataset dictionary keys
X_LSTG = 'x_lstg'
LOOKUP = 'lookup'

# filenames
COMPOSER_DIR = '{}composer/'.format(INPUT_DIR)
AGENT_FEATS_FILENAME = 'agent_feats.xlsx'
LOOKUP_FILENAME = 'lookup.gz'
X_LSTG_FILENAME = 'x_lstg.gz'
SIM_CHUNKS_DIR = 'chunks'
SIM_VALS_DIR = 'vals'
SIM_DISCRIM_DIR = 'discrim'

# temporal constants
MONTH = MAX_DAYS * 24 * 3600
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
START_TIME = 'start_time'
START_PRICE = 'start_price'
DEC_PRICE = 'decline_price'
ACC_PRICE = 'accept_price'

# meta categories with sale fees != .09 * price
META_7 = [21, 10]
META_6 = [32, 14, 11, 7, 28]


SELLER_HORIZON = 100
ENV_LSTG_COUNT = 1000

# space names
ACTION_SPACE_NAME = 'NegotiationActionSpace'
OBS_SPACE_NAME = 'NegotiationObsSpace'

# agents
INFO_AGENTS = ['byr', 'slr0', 'slr1']

# outcome tuple names
SALE = 'sale'
DUR = 'dur'
PRICE = 'price'

# param names
VAL_SE_CHECK = 50
MIN_SALES = 20
SIM_COUNT = 1
SE_TOLS = [.5, .75, 1.0, 1.25]
SE_RELAX_WIDTH = 10000
MAX_RECORDER_SIZE = 2e9

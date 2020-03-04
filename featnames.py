# clock feats
HOLIDAY = 'holiday'
DOW_PREFIX = 'dow'
TIME_OF_DAY = 'time_of_day'
AFTERNOON = 'afternoon'
CLOCK_FEATS = [HOLIDAY] + [DOW_PREFIX + str(i) for i in range(6)] + [TIME_OF_DAY, AFTERNOON]

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
DURATION = 'duration'
INT_REMAINING = 'remaining'

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
DAYS = 'days'
DELAY = 'delay'
CON = 'con'
NORM = 'norm'
SPLIT = 'split'
MSG = 'msg'
REJECT = 'reject'
AUTO = 'auto'
EXP = 'exp'

OUTCOME_FEATS = [DAYS, DELAY, AUTO, EXP, CON, REJECT, NORM, SPLIT, MSG]
ALL_OFFER_FEATS = CLOCK_FEATS + TIME_FEATS + OUTCOME_FEATS

# turn indices
TURN_FEATS = {
    'init_byr': ['t1', 't3', 't5'],
    'init_slr': ['t2', 't4']
}

# thread features
BYR_HIST = 'byr_hist'
MONTHS_SINCE_LSTG = 'months_since_lstg'
MONTHS_SINCE_LAST = 'months_since_last'

# lookup column names
LSTG = 'lstg'
SLR = 'slr'
STORE = 'store'
META = 'meta'
CAT = 'cat'
START_TIME = 'start_time'
START_PRICE = 'start_price'
DEC_PRICE = 'decline_price'
ACC_PRICE = 'accept_price'


# lstg feats
RELISTED = 'relisted'

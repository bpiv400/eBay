# strings for referencing quantities related to buyer and seller interface
SLR = 'slr'
BYR = 'byr'
ARRIVAL = 'arrival'

# hyperparameters
DELTA = 'delta'
DROPOUT = 'dropout'
ENTROPY = 'entropy'

# clock feats
HOLIDAY = 'holiday'
DOW_PREFIX = 'dow'
TIME_OF_DAY = 'time_of_day'
AFTERNOON = 'afternoon'

DATE_FEATS = [HOLIDAY] + [DOW_PREFIX + str(i) for i in range(6)]
CLOCK_FEATS = DATE_FEATS + [TIME_OF_DAY, AFTERNOON]

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
ACCEPT = 'accept'
REJECT = 'reject'
AUTO = 'auto'
EXP = 'exp'

OUTCOME_FEATS = [DAYS, DELAY, AUTO, EXP, CON, REJECT, NORM, SPLIT, MSG]
ALL_OFFER_FEATS = CLOCK_FEATS + TIME_FEATS + OUTCOME_FEATS

# thread features
BYR_HIST = 'byr_hist'
DAYS_SINCE_LSTG = 'days_since_lstg'
DAYS_SINCE_LAST = 'days_since_last'
BYR_AGENT = 'byr_agent'

# index labels
LSTG = 'lstg'
THREAD = 'thread'
INDEX = 'index'

# lookup column names
META = 'meta'
LEAF = 'leaf'
START_DATE = 'start_date'
START_TIME = 'start_time'
END_TIME = 'end_time'
START_PRICE = 'start_price'
DEC_PRICE = 'decline_price'
SLR_BO_CT = 'slr_bo_ct'
STORE = 'store'

# agents costs
MONTHLY_DISCOUNT = 'monthly_discount'
ACTION_DISCOUNT = 'action_discount'
ACTION_COST = 'action_cost'

# partition components
X_LSTG = 'x_lstg'
X_OFFER = 'x_offer'
X_THREAD = 'x_thread'
CLOCK = 'clock'
LOOKUP = 'lookup'
P_ARRIVAL = 'p_arrival'

# outcome types
SIM = 'sim'
OBS = 'obs'
RL = 'rl'

# turn feats
TURN_FEATS = {
    BYR: ['t1', 't3', 't5'],
    SLR: ['t2', 't4']
}

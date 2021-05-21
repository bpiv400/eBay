# strings for referencing quantities related to buyer and seller interface
SLR = 'slr'
BYR = 'byr'
ARRIVAL = 'arrival'

# hyperparameters
DROPOUT = 'dropout'
DELTA = 'delta'
TURN_COST = 'turn_cost'

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
COMMON = 'common'
MSG = 'msg'
ACCEPT = 'accept'
REJECT = 'reject'
AUTO = 'auto'
EXP = 'exp'

OUTCOME_FEATS = [DAYS, DELAY, AUTO, EXP, CON, REJECT, NORM, COMMON, MSG]
ALL_OFFER_FEATS = CLOCK_FEATS + TIME_FEATS + OUTCOME_FEATS

# thread features
BYR_HIST = 'byr_hist'
DAYS_SINCE_LSTG = 'days_since_lstg'
DAYS_SINCE_LAST = 'days_since_last'
IS_AGENT = 'is_agent'

# index labels
LSTG = 'lstg'
THREAD = 'thread'
INDEX = 'index'

# lookup column names
META = 'meta'
LEAF = 'leaf'
CNDTN = 'cndtn'
START_DATE = 'start_date'
START_TIME = 'start_time'
END_TIME = 'end_time'
START_PRICE = 'start_price'
DEC_PRICE = 'decline_price'
ACC_PRICE = 'accept_price'
FDBK_SCORE = 'fdbk_score'
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
ARRIVALS = 'arrivals'

# outcome types
SIM = 'sim'
OBS = 'obs'
RL = 'rl'

# turn feats
TURN_FEATS = {
    BYR: ['t1', 't3'],
    SLR: ['t2', 't4']
}

# partitions
TRAIN_MODELS = 'sim'
TRAIN_RL = 'rl'
VALIDATION = 'valid'
TEST = VALIDATION  # TODO: rename to 'testing' when using real testing data
PARTITIONS = [TRAIN_MODELS, TRAIN_RL, VALIDATION, TEST]
SIM_PARTITIONS = [TRAIN_MODELS, VALIDATION, TEST]
AGENT_PARTITIONS = [TRAIN_RL, VALIDATION, TEST]
HOLDOUT_PARTITIONS = [VALIDATION, TEST]

# model names
FIRST_ARRIVAL_MODEL = 'first_arrival'
INTERARRIVAL_MODEL = 'next_arrival'
BYR_HIST_MODEL = 'hist'
DISCRIM_MODEL = 'discrim'
PLACEBO_MODEL = 'placebo'

# model groups
ARRIVAL_MODELS = [FIRST_ARRIVAL_MODEL, INTERARRIVAL_MODEL, BYR_HIST_MODEL]
DELAY_MODELS = ['{}{}'.format(DELAY, i) for i in range(2, 8)]
CON_MODELS = ['{}{}'.format(CON, i) for i in range(1, 8)]
MSG_MODELS = ['{}{}'.format(MSG, i) for i in range(1, 7)]
OFFER_MODELS = DELAY_MODELS + CON_MODELS + MSG_MODELS
MODELS = ARRIVAL_MODELS + OFFER_MODELS
CENSORED_MODELS = [INTERARRIVAL_MODEL] + DELAY_MODELS
DISCRIM_MODELS = [DISCRIM_MODEL, PLACEBO_MODEL]

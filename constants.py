from featnames import SLR, BYR, TRAIN_MODELS, TRAIN_RL, VALIDATION

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
INTERVAL = int(5 * MINUTE)
INTERVAL_CT_TURN = int(MAX_DELAY_TURN / INTERVAL)
INTERVAL_CT_ARRIVAL = int(MAX_DELAY_ARRIVAL / INTERVAL)

# multiplier for concession
CON_MULTIPLIER = 100

# indices for byr and slr offers
IDX = {
    BYR: [1, 3, 5, 7],
    SLR: [2, 4, 6]
}

# number of chunks
NUM_CHUNKS = 1024

# simulation counts
OUTCOME_SIMS = 10
ARRIVAL_SIMS = 100

# features to drop from 'lstg' grouping for byr agent
BYR_DROP = ['lstg_ct', 'bo_ct',
            'auto_decline', 'has_decline',
            'auto_accept', 'has_accept']

# for precision issues
EPS = 1e-8

# number of concessions available to agent
NUM_COMMON_CONS = 6

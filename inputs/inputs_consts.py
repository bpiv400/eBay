from constants import HOUR, MINUTE, MAX_DELAY, HIST_QUANTILES, \
    CON_MULTIPLIER, ARRIVAL_MODELS, BYR_HIST_MODEL, DELAY_MODELS, \
    CON_MODELS, MSG_MODELS, INIT_POLICY_MODELS, INIT_VALUE_MODELS, \
    DISCRIM_MODELS, MAX_NORM_VALUE

# number of observations in small dataset
N_SMALL = 100000

# intervals for checking offer arrivals
INTERVAL = {1: HOUR,
            2: 5 * MINUTE,
            3: 30 * MINUTE,
            4: 5 * MINUTE,
            5: 30 * MINUTE,
            6: 5 * MINUTE,
            7: 5 * MINUTE}

INTERVAL_COUNTS = {i: int(MAX_DELAY[i] / INTERVAL[i]) for i in INTERVAL.keys()}

# size of model output
NUM_OUT = dict()
for m in ARRIVAL_MODELS[:-1]:
    NUM_OUT[m] = INTERVAL_COUNTS[1] + 1
NUM_OUT[BYR_HIST_MODEL] = HIST_QUANTILES
for m in DELAY_MODELS:
    turn = int(m[-1])
    NUM_OUT[m] = INTERVAL_COUNTS[turn] + 1
for m in CON_MODELS[:-1] + INIT_POLICY_MODELS:
    NUM_OUT[m] = CON_MULTIPLIER + 1
for m in INIT_VALUE_MODELS:
    NUM_OUT[m] = MAX_NORM_VALUE + 1
for m in [CON_MODELS[-1]] + MSG_MODELS + DISCRIM_MODELS:
    NUM_OUT[m] = 1

# monthly discount rate for agents
MONTHLY_DISCOUNT = 0.995

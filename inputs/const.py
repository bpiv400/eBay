from constants import HOUR, MINUTE, MAX_DELAY, HIST_QUANTILES, \
    CON_MULTIPLIER, ARRIVAL_MODELS, BYR_HIST_MODEL, DELAY_MODELS, \
    CON_MODELS, MSG_MODELS, DISCRIM_MODELS, INIT_VALUE_MODELS, \
    SLR_POLICY_INIT, SLR_DELAY_POLICY_INIT, BYR_POLICY_INIT, \
    BYR_DELAY_POLICY_INIT

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

# costs for slr agent
DELTA_MONTH = .995
DELTA_ACTION = 1.
C_ACTION = 0.

# size of model output
NUM_OUT = dict()
for m in ARRIVAL_MODELS[:-1] + ['arrival']:
    NUM_OUT[m] = INTERVAL_COUNTS[1] + 1
NUM_OUT[BYR_HIST_MODEL] = HIST_QUANTILES
for m in DELAY_MODELS:
    turn = int(m[-1])
    NUM_OUT[m] = INTERVAL_COUNTS[turn] + 1
for m in CON_MODELS[:-1] + INIT_VALUE_MODELS:
    NUM_OUT[m] = CON_MULTIPLIER + 1
for m in [CON_MODELS[-1]] + MSG_MODELS + DISCRIM_MODELS:
    NUM_OUT[m] = 1
for m in [SLR_POLICY_INIT, BYR_POLICY_INIT]:
    NUM_OUT[m] = CON_MULTIPLIER + 1
for m in [BYR_DELAY_POLICY_INIT, SLR_DELAY_POLICY_INIT]:
    NUM_OUT[m] = CON_MULTIPLIER + 2

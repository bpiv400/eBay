from constants import HOUR, MINUTE, MAX_DELAY, HIST_QUANTILES, \
    CON_MULTIPLIER, FIRST_ARRIVAL_MODEL, INTERARRIVAL_MODEL, \
    BYR_HIST_MODEL, DELAY_MODELS, CON_MODELS, MSG_MODELS, \
    DISCRIM_MODELS, INIT_VALUE_MODELS, SLR_POLICY_INIT, \
    SLR_DELAY_POLICY_INIT, BYR_DELAY_POLICY_INIT

# number of observations in small dataset
N_SMALL = 100000

# intervals for checking offer arrivals
ARRIVAL_INTERVAL = HOUR
DELAY_INTERVAL = 5 * MINUTE

ARRIVAL_INTERVAL_CT = int(MAX_DELAY[1] / ARRIVAL_INTERVAL)
DELAY_INTERVAL_CT = int(MAX_DELAY[2] / DELAY_INTERVAL)

# costs for slr agent
DELTA_MONTH = .995
DELTA_ACTION = 1.
C_ACTION = 0.

# size of model output
NUM_OUT = dict()
for m in [FIRST_ARRIVAL_MODEL, INTERARRIVAL_MODEL]:
    NUM_OUT[m] = ARRIVAL_INTERVAL_CT + 1
NUM_OUT[BYR_HIST_MODEL] = HIST_QUANTILES
for m in DELAY_MODELS:
    NUM_OUT[m] = DELAY_INTERVAL_CT + 1
for m in CON_MODELS[:-1] + INIT_VALUE_MODELS:
    NUM_OUT[m] = CON_MULTIPLIER + 1
for m in [CON_MODELS[-1]] + MSG_MODELS + DISCRIM_MODELS:
    NUM_OUT[m] = 1
for m in [SLR_POLICY_INIT, BYR_DELAY_POLICY_INIT]:
    NUM_OUT[m] = CON_MULTIPLIER + 1
NUM_OUT[SLR_DELAY_POLICY_INIT] = CON_MULTIPLIER + 2

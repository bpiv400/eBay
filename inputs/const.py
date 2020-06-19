from constants import HOUR, MINUTE, HIST_QUANTILES, \
    CON_MULTIPLIER, BYR_HIST_MODEL, DELAY_MODELS, \
    CON_MODELS, MSG_MODELS, DISCRIM_MODELS, INIT_VALUE_MODELS, \
    SLR_POLICY_INIT, SLR_DELAY_POLICY_INIT, BYR_DELAY_POLICY_INIT, \
    MAX_DELAY_TURN, MAX_DELAY_ARRIVAL,\
    FIRST_ARRIVAL_MODEL, INTERARRIVAL_MODEL

# number of observations in small dataset
N_SMALL = 100000

INTERVAL_TURN = int(5 * MINUTE)
INTERVAL_ARRIVAL = int(HOUR)

INTERVAL_CT_TURN = int(MAX_DELAY_TURN / INTERVAL_TURN)
INTERVAL_CT_ARRIVAL = int(MAX_DELAY_ARRIVAL / INTERVAL_ARRIVAL)

# costs for slr agent
DELTA_MONTH = .995
DELTA_ACTION = 1.
C_ACTION = 0.

# size of model output
NUM_OUT = dict()
for m in [FIRST_ARRIVAL_MODEL, INTERARRIVAL_MODEL]:
    NUM_OUT[m] = INTERVAL_CT_ARRIVAL + 1
NUM_OUT[BYR_HIST_MODEL] = HIST_QUANTILES
for m in DELAY_MODELS:
    NUM_OUT[m] = INTERVAL_CT_TURN + 1
for m in CON_MODELS[:-1] + INIT_VALUE_MODELS:
    NUM_OUT[m] = CON_MULTIPLIER + 1
for m in [CON_MODELS[-1]] + MSG_MODELS + DISCRIM_MODELS:
    NUM_OUT[m] = 1
for m in [SLR_POLICY_INIT, BYR_DELAY_POLICY_INIT]:
    NUM_OUT[m] = CON_MULTIPLIER + 1
NUM_OUT[SLR_DELAY_POLICY_INIT] = CON_MULTIPLIER + 2

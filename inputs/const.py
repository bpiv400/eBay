from constants import CON_MULTIPLIER, INTERVAL_CT_TURN, INTERVAL_CT_ARRIVAL
from featnames import DELAY_MODELS, CON_MODELS, MSG_MODELS, \
    FIRST_ARRIVAL_MODEL, INTERARRIVAL_MODEL, BYR_HIST_MODEL, DISCRIM_MODEL

# size of model output
NUM_OUT = {
    FIRST_ARRIVAL_MODEL: INTERVAL_CT_ARRIVAL + 1,
    INTERARRIVAL_MODEL: INTERVAL_CT_ARRIVAL + 1,
    BYR_HIST_MODEL: 3
}
for m in DELAY_MODELS:
    NUM_OUT[m] = INTERVAL_CT_TURN + 1
for m in CON_MODELS[:-1]:
    NUM_OUT[m] = CON_MULTIPLIER + 1
for m in [CON_MODELS[-1]] + [DISCRIM_MODEL] + MSG_MODELS:
    NUM_OUT[m] = 1

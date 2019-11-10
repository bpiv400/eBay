#TODO: Move all of this into env_consts

# arrival models
from constants import SLR_PREFIX, BYR_PREFIX

NUM_OFFERS = 'arrival'
BYR_HIST = 'hist'

# offer models
CON = 'cn'
DELAY = 'delay'
MSG = 'msg'

# prefixes
ARRIVAL_PREFIX = 'arrival'

# model sets
FEED_FORWARD = [DAYS, SEC]
ARRIVAL = FEED_FORWARD
RECURRENT = [DELAY, CON]
LSTM_MODELS = [DAYS, '{}_{}'.format(SLR_PREFIX, DELAY), '{}_{}'.format(BYR_PREFIX, DELAY)]
OFFER_NO_PREFIXES = [model for model in RECURRENT if model != DAYS]
MODELS_NO_PREFIXES = RECURRENT + FEED_FORWARD
OFFER = ['{}_{}'.format(SLR_PREFIX, model) for model in OFFER_NO_PREFIXES] + \
        ['{}_{}'.format(BYR_PREFIX, model) for model in OFFER_NO_PREFIXES]
MODELS = OFFER + FEED_FORWARD


# arrival models
from constants import SLR_PREFIX, BYR_PREFIX

DAYS = 'days'  # recurrent
SEC = 'sec'

# offer models
CON = 'cn'
DELAY = 'delay'

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


#TODO: Move all of this into env_consts

# arrival interface
from constants import SLR_PREFIX, BYR_PREFIX

NUM_OFFERS = 'arrival'
BYR_HIST = 'hist'

# offer interface
CON = 'cn'
DELAY = 'delay'
MSG = 'msg'

# prefixes
ARRIVAL_PREFIX = 'arrival'

# model sets
FEED_FORWARD = [BYR_HIST]
ARRIVAL = [NUM_OFFERS, BYR_HIST]
RECURRENT = [NUM_OFFERS, CON, DELAY, MSG]
LSTM_MODELS = [NUM_OFFERS, '{}_{}'.format(SLR_PREFIX, DELAY),
               '{}_{}'.format(BYR_PREFIX, DELAY)]
OFFER_NO_PREFIXES = [model for model in RECURRENT if model != NUM_OFFERS]
MODELS_NO_PREFIXES = RECURRENT + FEED_FORWARD
OFFER = ['{}_{}'.format(SLR_PREFIX, model) for model in OFFER_NO_PREFIXES] + \
        ['{}_{}'.format(BYR_PREFIX, model) for model in OFFER_NO_PREFIXES]
MODELS = OFFER + ARRIVAL


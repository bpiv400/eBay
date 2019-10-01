# arrival models
DAYS = 'days' # recurrent
BIN = 'bin'
HIST = 'hist'
LOC = 'loc'
SEC = 'sec'

# offer models
ACC = 'accept'
REJ = 'reject'
RND = 'round'
NINE = 'nines'
MSG = 'msg'
CON = 'cn'
DELAY = 'delay'

# prefixes
SLR_PREFIX = 'slr'
BYR_PREFIX = 'byr'
ARRIVAL_PREFIX = 'arrival'

# model sets
OFFER_INDS = []
FEED_FORWARD = [BIN, HIST, LOC, SEC]
RECURRENT = [DAYS, DELAY, ACC, REJ, RND, NINE, MSG, CON]
LSTM_MODELS = [DAYS, '{}_{}'.format(SLR_PREFIX, DELAY), '{}_{}'.format(BYR_PREFIX, DELAY)]
OFFER_NO_PREFIXES = [model for model in RECURRENT if model != DAYS]
ARRIVAL = FEED_FORWARD + [DAYS]
MODELS_NO_PREFIXES = RECURRENT + FEED_FORWARD

OFFER = ['{}_{}'.format(SLR_PREFIX, model) for model in OFFER_NO_PREFIXES] + \
        ['{}_{}'.format(BYR_PREFIX, model) for model in OFFER_NO_PREFIXES]
MODELS = OFFER + FEED_FORWARD + [DAYS]


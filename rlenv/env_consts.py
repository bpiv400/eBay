from utils import unpickle
from constants import INPUT_DIR, SLR_PREFIX, BYR_PREFIX
from featnames import CON, DELAY, MSG

INTERACT = False

# dataset dictionary keys
X_LSTG = 'x_lstg'
LOOKUP = 'lookup'
INTERVAL = 'interval'

# filenames
COMPOSER_DIR = '{}composer/'.format(INPUT_DIR)  # location of composer
AGENT_FEATS_FILENAME = 'agent_feats.xlsx'  # location of file containing lists of features for all agents
LOOKUP_FILENAME = 'lookup.gz'
X_LSTG_FILENAME = 'x_lstg.gz'
# partition subdir names
SIM_CHUNKS_DIR = 'chunks'
SIM_VALS_DIR = 'vals'
SIM_DISCRIM_DIR = 'discrim'

# fee constants
LISTING_FEE = .03

# meta categories with sale fees != .09 * price
META_7 = [21, 10]
META_6 = [32, 14, 11, 7, 28]

# holiday and day-of-week indicators, indexed by days since START
DATE_FEATS = unpickle(INPUT_DIR + 'date_feats.pkl')

# various counts
SELLER_HORIZON = 100
ENV_LSTG_COUNT = 1000

# space names
ACTION_SPACE_NAME = 'NegotiationActionSpace'
OBS_SPACE_NAME = 'NegotiationObsSpace'

# agent names (see AGENTS_FEATS_FILENAME)
INFO_AGENTS = ['byr', 'slr0', 'slr1']

# outcome tuple names
SALE = 'sale'
DUR = 'dur'
PRICE = 'price'

# param names
MIN_SALES = 20
SE_TOL = .5

# composer maps
SIZE = 'size'
LSTG_MAP = 'lstg'
THREAD_MAP = 'thread'
TURN_IND_MAP= 'turns'
X_TIME_MAP = 'x_time'

# offer response indicators
ACC_IND = 0
REJ_IND = 1
OFF_IND = 2

# lstg level
ARRIVAL = 'ARRIVAL'

# thread level
FIRST_OFFER = 'FIRST_OFFER' # first byer offer
SELLER_DELAY = 'seller_delay'
BUYER_DELAY = 'buyer_delay'

# offer level
BUYER_OFFER = 'buyer_offer'
SELLER_OFFER = 'seller_offer'

# model names
ARRIVAL_MODEL = 'arrival'
BYR_HIST_MODEL = 'hist'

# model sets
ARRIVAL_MODELS = [ARRIVAL_MODEL, BYR_HIST_MODEL]
LSTM_MODELS = [ARRIVAL_MODEL, '{}_{}'.format(DELAY, SLR_PREFIX), '{}_{}'.format(DELAY, BYR_PREFIX)]
RECURRENT_MODELS = LSTM_MODELS
OFFER_NO_PREFIXES = [CON, MSG, DELAY]
OFFER_MODELS = ['{}_{}'.format(model, SLR_PREFIX) for model in OFFER_NO_PREFIXES] + \
        ['{}_{}'.format(model, BYR_PREFIX) for model in OFFER_NO_PREFIXES]
FEED_FORWARD_MODELS = [model for model in OFFER_MODELS if model not in RECURRENT_MODELS]
FEED_FORWARD_MODELS.append(BYR_HIST_MODEL)
MODELS = OFFER_MODELS + ARRIVAL_MODELS

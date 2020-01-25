from utils import unpickle
from constants import INPUT_DIR, SLR_PREFIX, BYR_PREFIX
from featnames import (CON, DELAY, MSG, ALL_OFFER_FEATS, CLOCK_FEATS,
                       TIME_FEATS, DAYS, EXP, NORM, SPLIT, AUTO, REJECT)

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


# space names
ACTION_SPACE_NAME = 'NegotiationActionSpace'
OBS_SPACE_NAME = 'NegotiationObsSpace'

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
TURN_IND_MAP = 'turns'
CLOCK_MAP = 'clock'
TIME_MAP = 'time'
OFFER_MAPS = dict()
for i in range(1, 8):
    OFFER_MAPS[i] = 'offer{}'.format(i)

# offer response indicators
ACC_IND = 0
REJ_IND = 1
OFF_IND = 2

# lstg level
ARRIVAL = 'ARRIVAL'

# thread level
FIRST_OFFER = 'FIRST_OFFER' # first byer offer
OFFER_EVENT = 'OFFER'
DELAY_EVENT = 'DELAY'

# model names
ARRIVAL_MODEL = 'arrival'
BYR_HIST_MODEL = 'hist'

# model sets
ARRIVAL_MODELS = [ARRIVAL_MODEL, BYR_HIST_MODEL]
OFFER_NO_PREFIXES = [CON, MSG, DELAY]
OFFER_MODELS = ['{}_{}'.format(model, SLR_PREFIX) for model in OFFER_NO_PREFIXES] + \
        ['{}_{}'.format(model, BYR_PREFIX) for model in OFFER_NO_PREFIXES]
MODELS = OFFER_MODELS + ARRIVAL_MODELS


# useful indices in offer feats
CLOCK_START_IND = 0
CLOCK_END_IND = len(CLOCK_FEATS)

TIME_START_IND = len(CLOCK_FEATS)
TIME_END_IND = len(CLOCK_FEATS) + len(TIME_FEATS)

DELAY_START_IND = ALL_OFFER_FEATS.index(DAYS)
DELAY_END_IND = ALL_OFFER_FEATS.index(EXP) + 1

CON_START_IND = ALL_OFFER_FEATS.index(CON)

NORM_IND = ALL_OFFER_FEATS.index(NORM)
CON_IND = ALL_OFFER_FEATS.index(CON)
DELAY_IND = ALL_OFFER_FEATS.index(DELAY)
SPLIT_IND = ALL_OFFER_FEATS.index(SPLIT)
MSG_IND = ALL_OFFER_FEATS.index(MSG)
DAYS_IND = ALL_OFFER_FEATS.index(DAYS)
AUTO_IND = ALL_OFFER_FEATS.index(AUTO)
REJECT_IND = ALL_OFFER_FEATS.index(REJECT)
EXP_IND = ALL_OFFER_FEATS.index(EXP)


# Reinforcement learning environment parameters

# various counts
SELLER_HORIZON = 100
ENV_LSTG_COUNT = 1000

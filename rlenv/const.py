from utils import unpickle
from constants import DATE_FEATS_PATH
from featnames import CON, DELAY, MSG, ALL_OFFER_FEATS, CLOCK_FEATS, TIME_FEATS, \
    DAYS, EXP, NORM, COMMON, AUTO, REJECT, BYR_OFFERS_RECENT, THREAD_COUNT

INTERACT = False

# partition subdir names
SIM_CHUNKS_DIR = 'chunks'
SIM_VALS_DIR = 'vals'
SIM_DISCRIM_DIR = 'outcomes'

# holiday and day-of-week indicators, indexed by days since START
DATE_FEATS_DF = unpickle(DATE_FEATS_PATH)

# outcome tuple names
SALE = 'sale'
DUR = 'dur'
PRICE = 'price'

# param names

# composer maps
SIZE = 'size'
LSTG_MAP = 'lstg'
TURN_IND_MAP = 'turns'
CLOCK_MAP = 'clock'
TIME_MAP = 'time'
OFFER_MAPS = {t: 'offer{}'.format(t) for t in range(1, 8)}

# offer response indicators
ACC_IND = 0
REJ_IND = 1
OFF_IND = 2

# event types
EXPIRATION = 'EXPIRATION'
OFFER_EVENT = 'OFFER'
DELAY_EVENT = 'DELAY'

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
SPLIT_IND = ALL_OFFER_FEATS.index(COMMON)
MSG_IND = ALL_OFFER_FEATS.index(MSG)
DAYS_IND = ALL_OFFER_FEATS.index(DAYS)
AUTO_IND = ALL_OFFER_FEATS.index(AUTO)
REJECT_IND = ALL_OFFER_FEATS.index(REJECT)
EXP_IND = ALL_OFFER_FEATS.index(EXP)
BYR_OFFERS_RECENT_IND = ALL_OFFER_FEATS.index(BYR_OFFERS_RECENT)
THREAD_COUNT_IND = ALL_OFFER_FEATS.index(THREAD_COUNT)

# used to update time valued features after some event has occurred
SLR_REJECTION = 'SLR_REJECTION'  # slr rejecting an offer on turn 2 or turn 4
BYR_REJECTION = 'BYR_REJECTION'  # closes the thread
OFFER = 'OFFER'  # buyer or seller offer that's not an acceptance or rejection

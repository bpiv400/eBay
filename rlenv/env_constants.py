from utils import unpickle

SLR_OFFERS = 'slr_offers'
BYR_OFFERS = 'byr_offers'
SLR_OFFERS_OPEN = 'slr_offers_open'
BYR_OFFERS_OPEN = 'byr_offers_open'
SLR_BEST = 'slr_best'
BYR_BEST = 'byr_best'
SLR_BEST_OPEN = 'slr_best_open'
BYR_BEST_OPEN = 'byr_best_open'

TIME_FEATS = [
    SLR_OFFERS,
    BYR_OFFERS,
    SLR_OFFERS_OPEN,
    BYR_OFFERS_OPEN,
    SLR_BEST,
    BYR_BEST,
    SLR_BEST_OPEN,
    BYR_BEST_OPEN
]

EXPIRATION = 48 * 60 * 60

INPUT_DIR = 'data/partitions/train_rl'
LSTG_FILENAME = '{}/lstg.h5df'.format(INPUT_DIR)
COL_FILENAME = '{}/x_lstg_cols.pkl'.format(INPUT_DIR)
LSTG_COLS = unpickle(COL_FILENAME)
MONTH = 30 * 24 * 3600
DAY = 24 * 3600

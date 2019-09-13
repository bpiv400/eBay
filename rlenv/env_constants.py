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

DATA_DIR = 'data/partitions/train_rl'
MODEL_DIR = 'data/models'
LSTG_FILENAME = '{}/lstg.h5df'.format(DATA_DIR)
FIXED_COLS_FILENAME = '{}/x_lstg_cols.pkl'.format(DATA_DIR)
COMPOSER_FILENAME = '{}/composer.pkl'
LSTG_COLS = unpickle(FIXED_COLS_FILENAME)
INFO_FILENAME = 'info.pkl'
MONTH = 30 * 24 * 3600
DAY = 24 * 3600

ARRIVAL_CLOCK_FEATS = ['focal_days', 'focal_holiday', 'focal_dow0',
                       'focal_dow1', 'focal_dow2', 'focal_dow3',
                       'focal_dow4', 'focal_dow5']

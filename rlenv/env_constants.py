from collections import OrderedDict

# the list of columns contained in the 2d np.array containing all lstgs for a
# given seller
LSTG_COLUMNS = ['start_date', 'meta', 'leaf', 'cndtn', 'title', 'lstg', 'slr_hist',
                'relisted', 'fdbk_score', 'fdbk_pstv', 'start_price',
                'photos', 'slr_lstgs', 'slr_bos', 'decline_price', 'accept_price',
                'store', 'slr_us', 'fast', 'end_time']

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

def make_column_map(cols, consts=False):
    """
    Generates a dictionary mapping column name to position in numpy array
    of constants stored for environment input

    :return: OrderedDict
    """
    col_map = OrderedDict()
    for i, name in enumerate(cols):
        col_map[name] = i
    if consts:
        i = len(col_map)
        col_map['byr_us'] = i
        col_map['byr_hist'] = (i+1)
    return col_map


# ordered dictionary for accessing elements of const array
LSTG_FEATS_MAP = make_column_map(LSTG_COLUMNS, consts=False)
CONSTS_MAP = make_column_map(CONSTS, consts=True)

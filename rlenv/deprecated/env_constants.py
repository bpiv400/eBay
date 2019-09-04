from collections import OrderedDict

# the list of columns contained in the 2d np.array containing all lstgs for a
# given seller
LSTG_COLUMNS = ['start_date', 'meta', 'leaf', 'cndtn', 'title', 'lstg', 'slr_hist',
                'relisted', 'fdbk_score', 'fdbk_pstv', 'start_price',
                'photos', 'slr_lstgs', 'slr_bos', 'decline_price', 'accept_price',
                'store', 'slr_us', 'fast', 'end_time']

# all columns that correspond to an identifier, rather than a feature of the lstg
LSTG_IDS = ['slr', 'meta', 'leaf', 'cndtn', 'title', 'lstg']

# all columns except identifiers
CONSTS = [col for col in LSTG_COLUMNS if col not in LSTG_IDS]

# dictionary of lists where order defines feature hierarchy  from broadest to
# most narrow and each entry gives an idientifier level containing names of
# features to be computed at that level
TIME_FEATS = OrderedDict({
    'slr': [
        'open_lstgs',
        'open_threads'
        ],
    'meta': [
        'open_lstgs',
        'open_threads'
        ],
    'leaf': [
        'open_lstgs',
        'open_threads'
        ],
    'title': [
        'open_lstgs',
        'open_threads',
        'byr_offers',
        'slr_offers',
        ],
    'cndtn': [
        'open_lstgs',
        'open_threads',
        'byr_offers_recent',
        'slr_offers_recent',
        'slr_min',
        'byr_max',
        'byr_max_recent',
        'slr_min_recent'
        ],
    'lstg': [
        'open_threads',
        'byr_offers',
        'slr_offers',
        'slr_min',
        'byr_max',
        'byr_max_recent',
        'slr_min_recent'
        ]
})


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

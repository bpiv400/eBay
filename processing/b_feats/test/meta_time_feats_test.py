import pytest
from processing.b_feats.util import get_cat_time_feats
import numpy as np
import pandas as pd
from rlenv.time_triggers import *


COLS = ['accept', 'start_price', '']
index = ['meta', '']



def add_event(df, offer, trigger_type):
    """
    Adds an event to an events dataframe using the same dictionary of offer features
    as that TimeFeatures.update_feat
    :param df: dataframe containing events up to this point
    :param trigger_type: str defined in rlenv.time_triggers
    :param thread_id: int giving the id of the thread
    :param offer: dictionary containing price (norm), time, type (str indicating 'byr' or 'slr', thread_id
    :return: updated dataframe
    """
    offer = offer.copy()
    if df is None:
        df = pd.DataFrame(columns=COLS)
        df.index = pd.MultiIndex(levels=[[], [], []],
                                 codes=[[], [], []],
                                 names=['lstg', 'thread', 'index'])
    if lstg in df.index.levels[0]:
        last_index = df.xs(lstg, level='lstg', drop_level=True)
        if (lstg, thread_id, 1) in df.index:
            last_index = last_index.xs(thread_id, level='thread',
                                       drop_level=True).reset_index()['index'].max() + 1
        else:
            last_index = 1
    else:
        last_index = 1
    offer_index = pd.MultiIndex.from_tuples([(lstg, thread_id, last_index)],
                                            names=['lstg', 'thread', 'index'])

    # repurpose offer dictionary
    offer['start_price'] = 0
    offer['clock'] = offer['time']
    del offer['time']
    offer['byr'] = offer['type'] == 'byr'
    offer['accept'] = trigger_type == ACCEPTANCE
    offer['reject'] = trigger_type == BYR_REJECTION or trigger_type == SLR_REJECTION
    offer['norm'] = offer['price']
    for col in ['price', 'censored', 'message']:
        offer[col] = 0
    del offer['type']
    keys = list(offer.keys())
    keys.sort()
    cols = COLS.copy()
    cols.sort()
    assert len(keys) == len(cols)
    assert all([key == col for key, col in zip(keys, cols)])

    offer_df = pd.DataFrame(data=offer, index=offer_index)
    df = df.append(offer_df, verify_integrity=True, sort=True)
    return df



def setup_complex_lstg(events, meta=1, leaf=1):
    # start new lstg = 1

    # start new thread =1 for lstg = 1

    # slr counter offer in thread = 1 for lstg = 1

    # byr counter offer in thread = 1 for lstg = 1

    # wait 30 days

    # slr counter offer in thread = 1 for lstg = 1

    # byr rejection in thread = 1 for lstg = 1

    # start new thread = 2 for lstg = 1

    # slr auto reject for thread = 2

    # start new lstg = 2

    # start new thread = 3 for lstg = 1

    # byr offer in thread = 2 for lstg = 1

    # start new lstg = 3

    # slr counter offer in thread = 2 for lstg = 1

    # start new thread = 1 for lstg = 2

    # start new thread = 1 for lstg = 3

    # start new thread = 2 for lstg = 2

    # byr reject in thread = 2 for lstg = 1

    # close lstg = 1

    # wait 30 days

    # slr counter in thread = 1 for lstg = 3

    # slr counter in thread = 1 for lstg = 2

    # start new lstg = 4

    # byr counter in thread = 1 for lstg = 2

    # slr counter in thread = 2 for lstg = 2

    # byr accept in thread = 2 for lstg = 2 (simultaneous lstg close)

    # byr counter in thread = 1 for lstg = 3

    # start thread = 1 for lstg = 4

    # slr accept in thread = 1 for lstg = 3 (simultaneous lstg close)

    # slr accept in thread = 1 for lstg = 4 (simultanoues lstg close)

    # start lstg = 5

    # start thread = 1 for lstg = 5

    # start thread = 2 for lstg = 5

    # start lstg = 6

    # start thread = 1 for lstg = 5

    # start thread = 2 for lstg = 5

    # slr counter thread = 2 for lstg = 6

    # start lstg = 7

    # start thread = 1 for lstg = 7

    # slr accept in thread = 2 for lstg = 5

    # byr accept for thread = 2 for lstg = 6

    # slr reject for lstg = 7

    # byr accept for thread = 1 lstg = 7

    # start lstg = 8

    # start lstg = 9

    #  start thread = 1 for lstg = 9

    # slr accept thread = 1 for lstg = 9

    # start thread = 1 for lstg = 8

    # slr accept thread = 1 for lstg = 10




def test_lstgs_open_meta():
    pass

def test_lstgs_open_leaf():
    pass

def test_lstgs_meta():
    pass

def test_lstgs_leaf():
    pass

def test_slr_offers_meta():
    pass


def test_slr_offers_leaf():
    pass


def test_byr_offers_meta():
    pass


def test_slr_offers_leaf():
    pass


def test_slr_offers_meta():
    pass


def test_threads_meta():
    pass


def test_threads_leaf():
    pass


def test_accepts_meta():
    pass


def test_accepts_leaf():
    pass


def test_price_quantile_meta():
    pass


def test_price_quantile_leaf():
    pass
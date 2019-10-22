import pytest
import numpy as np
from copy import deepcopy
import pandas as pd
from rlenv.time_triggers import *
from processing.b_feats.util import get_cat_time_feats


COLS = ['accept', 'censored', 'clock', 'price', 'reject', 'byr', 'flag', 'start_price']
INDEX_LEVELS = ['meta', 'leaf', 'cndtn', 'lstg', 'thread', 'index']
EMPTIES = [[] for _ in INDEX_LEVELS]
NEW_LSTG = 'new_lstg'
ACCEPTANCE = 'accept'
EXPIRE_LSTG = 'expire_lstg'
CENSORED = 'censored'
CNDTN = 1


def add_event(df, offer, trigger_type=None, meta=None, leaf=None):
    """
    Adds an event to an events dataframe using the same dictionary of offer features
    as that TimeFeatures.update_feat
    :param df: dataframe containing events up to this point
    :param trigger_type: str defined in rlenv.time_triggers
    :param offer: dictionary containing 'price', 'clock', 'lstg', 'thread', 'byr'
    :return: updated dataframe
    """
    thread_based = trigger_type in [SLR_REJECTION, BYR_REJECTION, OFFER, ACCEPTANCE]
    #if thread_based:
    #    print('{}: (lstg: {}, thread: {})'.format(trigger_type,
    #                                              offer['lstg'],
    #                                              offer['thread']))
    #else:
    #    print('{}: (lstg: {})'.format(trigger_type, offer['lstg']))
    # print(df)
    offer = offer.copy()
    data = dict()
    if df is None:
        df = pd.DataFrame(columns=COLS)
        df.index = pd.MultiIndex(levels=deepcopy(EMPTIES),
                                 codes=deepcopy(EMPTIES),
                                 names=INDEX_LEVELS)
    if meta in df.index.levels[0]:
        last_index = df.xs(meta, level='meta', drop_level=True)
        if leaf in last_index.index.get_level_values('leaf'):
            last_index = last_index.xs(leaf, level='leaf', drop_level=True)
            last_index = last_index.xs(CNDTN, level='cndtn', drop_level=True)
            if trigger_type == NEW_LSTG:
                # new lstg
                last_index = 0
                offer['thread'] = 0
            elif trigger_type == EXPIRE_LSTG:
                # lstg expiration
                last_index = 1
                offer['thread'] = 0
            else:
                last_index = last_index.xs(offer['lstg'], level='lstg',
                                           drop_level=True)
                if offer['thread'] in last_index.index.get_level_values('thread'):
                    last_index = last_index.xs(offer['thread'], level='thread', drop_level=True)
                    last_index = last_index.reset_index()['index'].max() + 1
                else:
                    # new thread
                    last_index = 1
        else:
            # new leaf and new lstg
            offer['thread'] = 0
            last_index = 0
    else:
        # new meta, leaf, and lstg
        offer['thread'] = 0
        last_index = 0

    offer_index = pd.MultiIndex.from_tuples([(meta, leaf, CNDTN, offer['lstg'], offer['thread'],
                                              last_index)], names=INDEX_LEVELS)
    # print('index: {}'.format(offer_index))

    # repurpose offer dictionary
    data['start_price'] = 100
    data['flag'] = False
    data['clock'] = offer['clock']
    data['accept'] = trigger_type == ACCEPTANCE
    data['censored'] = trigger_type == CENSORED
    data['reject'] = trigger_type == CENSORED or \
        trigger_type == BYR_REJECTION or trigger_type == SLR_REJECTION

    if trigger_type == NEW_LSTG or trigger_type == EXPIRE_LSTG:
        data['price'] = 0
        data['byr'] = False
    else:
        data['price'] = offer['price']
        data['byr'] = offer['byr']

    # prepare for row creation
    keys = list(data.keys())
    keys.sort()
    cols = COLS.copy()
    cols.sort()
    assert len(keys) == len(cols)
    assert all([key == col for key, col in zip(keys, cols)])

    # create row
    offer_df = pd.DataFrame(data=data, index=offer_index)

    df = df.append(offer_df, verify_integrity=True, sort=True)
    # print('contains 4: {}'.format((1, 1, 1, 4, 0, 0) in df.index))
    return df


def setup_complex_lstg(events, meta=1, leaf=1, starter=0):
    # start new lstg = 1
    offer = {
        'lstg': 1 + starter,
        'clock': 15,
    }
    events = add_event(events, offer, trigger_type=NEW_LSTG, meta=meta, leaf=leaf)

    # start new thread =1 for lstg = 1
    offer['price'] = 50
    offer['thread'] = 1
    offer['byr'] = True
    offer['clock'] = 20
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # slr counter offer in thread = 1 for lstg = 1
    offer['byr'] = False
    offer['price'] = 80
    offer['clock'] = 30
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # byr counter offer in thread = 1 for lstg = 1
    offer['byr'] = True
    offer['price'] = 60
    offer['clock'] = 40
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # wait 30 days

    # slr counter offer in thread = 1 for lstg = 1
    offer['byr'] = False
    offer['price'] = 75
    offer['clock'] = 70
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # byr rejection in thread = 1 for lstg = 1
    offer['byr'] = True
    offer['price'] = 60
    offer['clock'] = 75
    events = add_event(events, offer, trigger_type=BYR_REJECTION, meta=meta, leaf=leaf)

    # start new thread = 2 for lstg = 1
    offer['thread'] = 2
    offer['price'] = 30
    offer['clock'] = 80
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # slr auto reject for thread = 2
    offer['byr'] = False
    offer['price'] = 100
    events = add_event(events, offer, trigger_type=SLR_REJECTION, meta=meta, leaf=leaf)

    # start new lstg = 2
    offer['lstg'] = 2 + starter
    offer['clock'] = 90
    events = add_event(events, offer, trigger_type=NEW_LSTG, meta=meta, leaf=leaf)

    # start new thread = 3 for lstg = 1
    offer['thread'] = 3
    offer['price'] = 50
    offer['lstg'] = 1 + starter
    offer['byr'] = True
    offer['clock'] = 100
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # byr offer in thread = 2 for lstg = 1
    offer['thread'] = 2
    offer['clock'] = 110
    offer['price'] = 70
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # start new lstg = 3
    offer['lstg'] = 3 + starter
    offer['clock'] = 115
    events = add_event(events, offer, trigger_type=NEW_LSTG, meta=meta, leaf=leaf)

    # slr counter offer in thread = 2 for lstg = 1
    offer['byr'] = False
    offer['price'] = 75
    offer['thread'] = 2
    offer['lstg'] = 1 + starter
    offer['clock'] = 120
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # start new thread = 1 for lstg = 2
    offer['lstg'] = 2 + starter
    offer['thread'] = 1
    offer['clock'] = 125
    offer['price'] = 50
    offer['byr'] = True
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # start new thread = 1 for lstg = 3
    offer['lstg'] = 3 + starter
    offer['thread'] = 1
    offer['clock'] = 125
    offer['price'] = 50
    offer['byr'] = True
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # start new thread = 2 for lstg = 2
    offer['lstg'] = 2 + starter
    offer['thread'] = 2
    offer['clock'] = 130
    offer['price'] = 55
    offer['byr'] = True
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # byr reject in thread = 2 for lstg = 1
    offer['lstg'] = 1 + starter
    offer['thread'] = 2
    offer['clock'] = 135
    offer['price'] = 70
    offer['byr'] = True
    events = add_event(events, offer, trigger_type=BYR_REJECTION, meta=meta, leaf=leaf)

    # close lstg = 1
    offer['lstg'] = 1 + starter
    offer['clock'] = 140
    events = add_event(events, offer, trigger_type=EXPIRE_LSTG, meta=meta, leaf=leaf)

    # wait 30 days

    # slr counter in thread = 1 for lstg = 3
    offer['byr'] = False
    offer['clock'] = 145
    offer['price'] = 75
    offer['lstg'] = 3 + starter
    offer['thread'] = 1
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # slr counter in thread = 1 for lstg = 2
    offer['byr'] = False
    offer['clock'] = 145
    offer['price'] = 80
    offer['lstg'] = 2 + starter
    offer['thread'] = 1
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # start new lstg = 4
    offer['lstg'] = 4 + starter
    offer['clock'] = 145
    events = add_event(events, offer, trigger_type=NEW_LSTG, meta=meta, leaf=leaf)

    # byr counter in thread = 1 for lstg = 2
    offer['byr'] = True
    offer['clock'] = 150
    offer['price'] = 60
    offer['lstg'] = 2 + starter
    offer['thread'] = 1
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # slr counter in thread = 2 for lstg = 2
    offer['byr'] = False
    offer['clock'] = 155
    offer['price'] = 80
    offer['lstg'] = 2 + starter
    offer['thread'] = 2
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # byr accept in thread = 2 for lstg = 2 (simultaneous lstg close)
    offer['byr'] = True
    offer['clock'] = 160
    offer['price'] = 80
    offer['lstg'] = 2 + starter
    offer['thread'] = 2
    events = add_event(events, offer, trigger_type=ACCEPTANCE, meta=meta, leaf=leaf)
    offer['byr'] = False
    offer['price'] = 80
    offer['lstg'] = 2 + starter
    offer['thread'] = 1
    events = add_event(events, offer, trigger_type=CENSORED, meta=meta, leaf=leaf)

    # byr counter in thread = 1 for lstg = 3
    offer['byr'] = True
    offer['clock'] = 165
    offer['price'] = 70
    offer['lstg'] = 3 + starter
    offer['thread'] = 1
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # start thread = 1 for lstg = 4
    offer['byr'] = True
    offer['clock'] = 170
    offer['price'] = 50
    offer['lstg'] = 4 + starter
    offer['thread'] = 1
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # slr accept in thread = 1 for lstg = 3 (simultaneous lstg close)
    offer['byr'] = False
    offer['clock'] = 175
    offer['price'] = 70
    offer['lstg'] = 3 + starter
    offer['thread'] = 1
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # slr accept in thread = 1 for lstg = 4 (simultanoues lstg close)
    offer['byr'] = False
    offer['clock'] = 180
    offer['price'] = 50
    offer['lstg'] = 4 + starter
    offer['thread'] = 1
    events = add_event(events, offer, trigger_type=ACCEPTANCE, meta=meta, leaf=leaf)

    # start lstg = 5
    offer['clock'] = 185
    offer['lstg'] = 5 + starter
    events = add_event(events, offer, trigger_type=NEW_LSTG, meta=meta, leaf=leaf)

    # start thread = 1 for lstg = 5
    offer['clock'] = 190
    offer['lstg'] = 5 + starter
    offer['thread'] = 1
    offer['price'] = 75
    offer['byr'] = True
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # start thread = 2 for lstg = 5
    offer['clock'] = 195
    offer['lstg'] = 5 + starter
    offer['thread'] = 2
    offer['price'] = 60
    offer['byr'] = True
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # start lstg = 6
    offer['clock'] = 200
    offer['lstg'] = 6 + starter
    events = add_event(events, offer, trigger_type=NEW_LSTG, meta=meta, leaf=leaf)

    # start thread = 1 for lstg = 6
    offer['clock'] = 205
    offer['lstg'] = 6 + starter
    offer['thread'] = 1
    offer['price'] = 20
    offer['byr'] = True
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # start thread = 2 for lstg = 6
    offer['clock'] = 205
    offer['lstg'] = 6 + starter
    offer['thread'] = 2
    offer['price'] = 25
    offer['byr'] = True
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # slr counter thread = 2 for lstg = 6
    offer['clock'] = 210
    offer['lstg'] = 6 + starter
    offer['thread'] = 2
    offer['price'] = 40
    offer['byr'] = False
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # start lstg = 7
    offer['clock'] = 215
    offer['lstg'] = 7 + starter
    events = add_event(events, offer, trigger_type=NEW_LSTG, meta=meta, leaf=leaf)

    # start thread = 1 for lstg = 7
    offer['clock'] = 220
    offer['lstg'] = 7 + starter
    offer['thread'] = 1
    offer['price'] = 60
    offer['byr'] = True
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # slr accept in thread = 2 for lstg = 5
    offer['clock'] = 225
    offer['lstg'] = 5 + starter
    offer['thread'] = 2
    offer['price'] = 60
    offer['byr'] = False
    events = add_event(events, offer, trigger_type=ACCEPTANCE, meta=meta, leaf=leaf)
    offer['thread'] = 1
    offer['lstg'] = 5 + starter
    offer['byr'] = False
    offer['price'] = 100
    events = add_event(events, offer, trigger_type=CENSORED, meta=meta, leaf=leaf)

    # byr accept for thread = 2 for lstg = 6
    offer['clock'] = 230
    offer['lstg'] = 6 + starter
    offer['thread'] = 2
    offer['price'] = 25
    offer['byr'] = True
    events = add_event(events, offer, trigger_type=ACCEPTANCE, meta=meta, leaf=leaf)
    offer['thread'] = 1
    offer['lstg'] = 6 + starter
    offer['byr'] = False
    offer['price'] = 100
    events = add_event(events, offer, trigger_type=CENSORED, meta=meta, leaf=leaf)

    # slr reject for lstg = 7
    offer['clock'] = 235
    offer['lstg'] = 7 + starter
    offer['thread'] = 1
    offer['price'] = 100
    offer['byr'] = False
    events = add_event(events, offer, trigger_type=SLR_REJECTION, meta=meta, leaf=leaf)

    # byr accept for thread = 1 lstg = 7
    offer['clock'] = 240
    offer['lstg'] = 7 + starter
    offer['thread'] = 2
    offer['price'] = 100
    offer['byr'] = True
    events = add_event(events, offer, trigger_type=ACCEPTANCE, meta=meta, leaf=leaf)

    # start lstg = 8
    offer['clock'] = 255
    offer['lstg'] = 8 + starter
    events = add_event(events, offer, trigger_type=NEW_LSTG, meta=meta, leaf=leaf)

    # start thread = 1 for lstg = 8
    offer['clock'] = 260
    offer['lstg'] = 8 + starter
    offer['thread'] = 1
    offer['price'] = 90
    offer['byr'] = True
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # start lstg = 9
    offer['clock'] = 265
    offer['lstg'] = 9 + starter
    events = add_event(events, offer, trigger_type=NEW_LSTG, meta=meta, leaf=leaf)

    # start thread 1 for lstg = 9
    offer['clock'] = 267
    offer['lstg'] = 9 + starter
    offer['thread'] = 1
    offer['price'] = 30
    offer['byr'] = True
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # slr counter in thread = 1 for lstg = 9
    offer['clock'] = 268
    offer['lstg'] = 9 + starter
    offer['thread'] = 1
    offer['price'] = 85
    offer['byr'] = False
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # start lstg = 10
    offer['clock'] = 270
    offer['lstg'] = 10 + starter
    events = add_event(events, offer, trigger_type=NEW_LSTG, meta=meta, leaf=leaf)

    # byr accept thread = 1 for lstg = 9
    offer['clock'] = 275
    offer['lstg'] = 9 + starter
    offer['thread'] = 1
    offer['price'] = 85
    offer['byr'] = True
    events = add_event(events, offer, trigger_type=ACCEPTANCE, meta=meta, leaf=leaf)

    # slr counter in thread = 1 for lstg = 8
    offer['clock'] = 280
    offer['lstg'] = 8 + starter
    offer['thread'] = 1
    offer['price'] = 98
    offer['byr'] = False
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # byr counter in thread = 1 for lstg = 8
    offer['clock'] = 285
    offer['lstg'] = 8 + starter
    offer['thread'] = 1
    offer['price'] = 95
    offer['byr'] = False
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # slr accept thread = 1 for lstg = 8
    offer['clock'] = 285
    offer['lstg'] = 8 + starter
    offer['thread'] = 1
    offer['price'] = 95
    offer['byr'] = True
    events = add_event(events, offer, trigger_type=ACCEPTANCE, meta=meta, leaf=leaf)

    # start lstg 11
    offer['clock'] = 290
    offer['lstg'] = 11 + starter
    events = add_event(events, offer, trigger_type=NEW_LSTG, meta=meta, leaf=leaf)

    # close lstg 10
    offer['clock'] = 295
    offer['lstg'] = 10 + starter
    events = add_event(events, offer, trigger_type=EXPIRE_LSTG, meta=meta, leaf=leaf)

    # close lstg 11
    offer['clock'] = 295
    offer['lstg'] = 11 + starter
    events = add_event(events, offer, trigger_type=EXPIRE_LSTG, meta=meta, leaf=leaf)
    events.byr = events.byr.astype(bool)
    events.reject = events.reject.astype(bool)
    events.accept = events.accept.astype(bool)
    events.censored = events.censored.astype(bool)
    events.price = events.price.astype(np.int64)
    events.clock = events.price.astype(np.int64)
    return events


def test_lstgs_open_meta():
    events = setup_complex_lstg(None, meta=1, leaf=1)
    events = setup_complex_lstg(events, meta=2, leaf=1, starter=11)
    # check values
    print(events)
    # time feats
    get_cat_time_feats(events, levels=['meta'])



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


def test_byr_offers_leaf():
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

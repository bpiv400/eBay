import numpy as np
from copy import deepcopy
import pandas as pd
from time.offer_types import *
from processing.c_feats.util import get_all_cat_feats
from constants import MAX_DELAY


COLS = ['accept', 'censored', 'clock', 'price', 'reject', 'byr', 'flag', 'start_price',
        'start_price_pctile', 'arrival_rate', 'byr_hist']
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
    if 'byr_hist' not in offer:
        offer['byr_hist'] = 0
    if trigger_type == NEW_LSTG:
        if 'start_price_pctile' not in offer:
            offer['start_price_pctile'] = 0
        if 'arrival_rate' not in offer:
            offer['arrival_rate'] = 0
        data['start_price_pctile'] = offer['start_price_pctile']
        data['arrival_rate'] = offer['arrival_rate']
    else:
        data['arrival_rate'] = 0
        data['start_price_pctile'] = 0
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
        data['byr_hist'] = 0
    else:
        data['price'] = offer['price']
        data['byr'] = offer['byr']
        data['byr_hist'] = offer['byr_hist']

    # prepare for row creation
    keys = list(data.keys())
    keys.sort()
    cols = COLS.copy()
    cols.sort()
    assert len(keys) == len(cols)
    assert all([key == col for key, col in zip(keys, cols)])

    # create row
    offer_df = pd.DataFrame(data=data, index=offer_index)
    pre = df.copy(deep=True)
    df = df.append(offer_df, verify_integrity=True)
    return df


def add_bins(events, meta=1, leaf=1, starter=0):
    offer = {
        'lstg': 1 + starter,
        'clock': 300,
    }
    return events

def setup_complex_lstg(events, meta=1, leaf=1, starter=0):
    # start new lstg = 1
    offer = {
        'lstg': 1 + starter,
        'clock': 15,
        'start_price_pctile': .8,
        'arrival_rate': .4
    }
    events = add_event(events, offer, trigger_type=NEW_LSTG, meta=meta, leaf=leaf)

    # start new thread =1 for lstg = 1
    offer['price'] = 50
    offer['thread'] = 1
    offer['byr'] = True
    offer['byr_hist'] = .2
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
    offer['byr_hist'] = .9
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # slr auto reject for thread = 2
    offer['byr'] = False
    offer['price'] = 100
    events = add_event(events, offer, trigger_type=SLR_REJECTION, meta=meta, leaf=leaf)

    # start new lstg = 2
    offer['lstg'] = 2 + starter
    offer['clock'] = 90
    offer['start_price_pctile'] = .7
    offer['arrival_rate'] = .2
    events = add_event(events, offer, trigger_type=NEW_LSTG, meta=meta, leaf=leaf)

    # start new thread = 3 for lstg = 1
    offer['thread'] = 3
    offer['price'] = 50
    offer['lstg'] = 1 + starter
    offer['byr'] = True
    offer['clock'] = 100
    offer['byr_hist'] = .6
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # byr offer in thread = 2 for lstg = 1
    offer['thread'] = 2
    offer['clock'] = 110
    offer['price'] = 70
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # start new lstg = 3
    offer['lstg'] = 3 + starter
    offer['clock'] = 115
    offer['start_price_pctile'] = .9
    offer['arrival_rate'] = .3
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
    offer['byr_hist'] = .7
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # start new thread = 1 for lstg = 3
    offer['lstg'] = 3 + starter
    offer['thread'] = 1
    offer['clock'] = 125
    offer['price'] = 50
    offer['byr'] = True
    offer['byr_hist'] = .15
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # start new thread = 2 for lstg = 2
    offer['lstg'] = 2 + starter
    offer['thread'] = 2
    offer['clock'] = 130
    offer['price'] = 55
    offer['byr'] = True
    offer['byr_hist'] = .25
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
    offer['start_price_pctile'] = .4
    offer['arrival_rate'] = .6
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
    offer['byr_hist'] = .75
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # slr accept in thread = 1 for lstg = 3 (simultaneous lstg close)
    offer['byr'] = False
    offer['clock'] = 175
    offer['price'] = 70
    offer['lstg'] = 3 + starter
    offer['thread'] = 1
    events = add_event(events, offer, trigger_type=ACCEPTANCE, meta=meta, leaf=leaf)

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
    offer['start_price_pctile'] = .6
    offer['arrival_rate'] = .65
    events = add_event(events, offer, trigger_type=NEW_LSTG, meta=meta, leaf=leaf)

    # start thread = 1 for lstg = 5
    offer['clock'] = 190
    offer['lstg'] = 5 + starter
    offer['thread'] = 1
    offer['price'] = 75
    offer['byr'] = True
    offer['byr_hist'] = .5
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # start thread = 2 for lstg = 5
    offer['clock'] = 195
    offer['lstg'] = 5 + starter
    offer['thread'] = 2
    offer['price'] = 60
    offer['byr'] = True
    offer['byr_hist'] = .65
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # start lstg = 6
    offer['clock'] = 200
    offer['lstg'] = 6 + starter
    offer['start_price_pctile'] = .75
    offer['arrival_rate'] = .7
    events = add_event(events, offer, trigger_type=NEW_LSTG, meta=meta, leaf=leaf)

    # start thread = 1 for lstg = 6
    offer['clock'] = 205
    offer['lstg'] = 6 + starter
    offer['thread'] = 1
    offer['price'] = 20
    offer['byr'] = True
    offer['byr_hist'] = .25
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # start thread = 2 for lstg = 6
    offer['clock'] = 205
    offer['lstg'] = 6 + starter
    offer['thread'] = 2
    offer['price'] = 25
    offer['byr'] = True
    offer['byr_hist'] = .3
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
    offer['start_price_pctile'] = .2
    offer['arrival_rate'] = .45
    events = add_event(events, offer, trigger_type=NEW_LSTG, meta=meta, leaf=leaf)

    # start thread = 1 for lstg = 7
    offer['clock'] = 220
    offer['lstg'] = 7 + starter
    offer['thread'] = 1
    offer['price'] = 60
    offer['byr'] = True
    offer['byr_hist'] = .8
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
    offer['thread'] = 1
    offer['price'] = 100
    offer['byr'] = True
    events = add_event(events, offer, trigger_type=ACCEPTANCE, meta=meta, leaf=leaf)

    # start lstg = 8
    offer['clock'] = 255
    offer['lstg'] = 8 + starter
    offer['start_price_pctile'] = .3
    offer['arrival_rate'] = .35
    events = add_event(events, offer, trigger_type=NEW_LSTG, meta=meta, leaf=leaf)

    # start thread = 1 for lstg = 8
    offer['clock'] = 260
    offer['lstg'] = 8 + starter
    offer['thread'] = 1
    offer['price'] = 90
    offer['byr'] = True
    offer['byr_hist'] = .85
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # start lstg = 9
    offer['clock'] = 265
    offer['lstg'] = 9 + starter
    offer['start_price_pctile'] = .50
    offer['arrival_rate'] = .55
    events = add_event(events, offer, trigger_type=NEW_LSTG, meta=meta, leaf=leaf)

    # start thread 1 for lstg = 9
    offer['clock'] = 267
    offer['lstg'] = 9 + starter
    offer['thread'] = 1
    offer['price'] = 30
    offer['byr'] = True
    offer['byr_hist'] = .95
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
    offer['start_price_pctile'] = .65
    offer['arrival_rate'] = .8
    events = add_event(events, offer, trigger_type=NEW_LSTG, meta=meta, leaf=leaf)

    # byr accept thread = 1 for lstg = 9
    offer['clock'] = 275
    offer['lstg'] = 9 + starter
    offer['thread'] = 1
    offer['price'] = 85
    offer['byr'] = True
    events = add_event(events, offer, trigger_type=ACCEPTANCE, meta=meta, leaf=leaf)
    # print(events.tail())

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
    offer['byr'] = True
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
    offer['start_price_pctile'] = .1
    offer['arrival_rate'] = .9
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
    events.flag = events.flag.astype(bool)
    events.price = events.price.astype(np.int64)
    events.clock = events.clock.astype(np.int64)
    return events


def make_exp(exp, exp_array):
    if exp is None:
        exp = pd.DataFrame(data={'exp': exp_array}, index=np.arange(1, len(exp_array) + 1))
    else:
        start = len(exp.index) + 1
        new_index = pd.Index(np.arange(start, start + len(exp_array)))
        new_exp = pd.DataFrame(data={'exp': exp_array}, index=new_index)
        exp = exp.append(new_exp, verify_integrity=True, sort=True)
        exp = exp.sort_index()
    return exp

def test_lstgs_meta():
    events = setup_complex_lstg(None, meta=1, leaf=1)
    events = setup_complex_lstg(events, meta=2, leaf=1, starter=11)
    events.index = events.index.droplevel(['leaf', 'cndtn'])
    actual = get_all_cat_feats(events, levels=['meta'])
    exp_array = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    exp = make_exp(None, exp_array)
    exp = make_exp(exp, exp_array)['exp']
    assert np.all(np.isclose(exp.values, actual['meta_lstgs'].values))


def test_lstgs_leaf():
    events = setup_complex_lstg(None, meta=1, leaf=2)
    events = setup_complex_lstg(events, meta=1, leaf=1, starter=11)
    events.index = events.index.droplevel(['cndtn'])
    actual = get_all_cat_feats(events, levels=['meta', 'leaf'])
    exp_array = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    exp = make_exp(None, exp_array)
    exp = make_exp(exp, exp_array)['exp']
    assert np.all(np.isclose(exp.values, actual['leaf_lstgs'].values))
    exp_array = [21] * 22
    assert np.all(np.isclose(make_exp(None, exp_array),
                             actual['meta_lstgs'].values))
    events = setup_complex_lstg(None, meta=2, leaf=1)
    events = setup_complex_lstg(events, meta=1, leaf=1, starter=11)
    events.index = events.index.droplevel(['cndtn'])
    actual = get_all_cat_feats(events, levels=['meta', 'leaf'])
    assert np.all(np.isclose(exp.values, actual['leaf_lstgs'].values))
    assert np.all(np.isclose(exp.values, actual['meta_lstgs'].values))


def test_slr_offers_meta():
    events = setup_complex_lstg(None, meta=1, leaf=1)
    events = setup_complex_lstg(events, meta=2, leaf=1, starter=11)
    events.index = events.index.droplevel(['leaf', 'cndtn'])
    actual = get_all_cat_feats(events, levels=['meta'])
    exp_array = [6, 7, 8, 9, 9, 8, 9, 8, 8, 9, 9]
    exp_array = [curr_exp / 10 for curr_exp in exp_array]
    exp = make_exp(None, exp_array)
    exp = make_exp(exp, exp_array)['exp']
    print(exp.values)
    print(actual['meta_slr_offers'].values)
    assert np.all(np.isclose(exp.values, actual['meta_slr_offers'].values))


def test_slr_offers_leaf():
    events = setup_complex_lstg(None, meta=1, leaf=2)
    events = setup_complex_lstg(events, meta=1, leaf=1, starter=11)
    events.index = events.index.droplevel(['cndtn'])
    actual = get_all_cat_feats(events, levels=['meta', 'leaf'])
    exp_array = [6, 7, 8, 9, 9, 8, 9, 8, 8, 9, 9]
    exp_array = [curr_exp / 10 for curr_exp in exp_array]
    exp = make_exp(None, exp_array)
    exp = make_exp(exp, exp_array)['exp']
    assert np.all(np.isclose(exp.values, actual['leaf_slr_offers'].values))

    exp_array = [6, 7, 8, 9, 9, 8, 9, 8, 8, 9, 9]
    diff = [9 - ent for ent in exp_array]
    exp_array = [18 - ent_diff for ent_diff in diff]
    exp_array = [curr_exp / 21 for curr_exp in exp_array]
    exp = make_exp(None, exp_array)
    exp = make_exp(exp, exp_array)['exp']
    assert np.all(np.isclose(exp, actual['meta_slr_offers'].values))

    events = setup_complex_lstg(None, meta=2, leaf=1)
    events = setup_complex_lstg(events, meta=1, leaf=1, starter=11)
    events.index = events.index.droplevel(['cndtn'])
    actual = get_all_cat_feats(events, levels=['meta', 'leaf'])
    exp_array = [6, 7, 8, 9, 9, 8, 9, 8, 8, 9, 9]
    exp_array = [curr_exp / 10 for curr_exp in exp_array]
    exp = make_exp(None, exp_array)
    exp = make_exp(exp, exp_array)['exp']
    assert np.all(np.isclose(exp.values, actual['leaf_slr_offers'].values))
    assert np.all(np.isclose(exp.values, actual['meta_slr_offers'].values))


def test_byr_offers_meta():
    events = setup_complex_lstg(None, meta=1, leaf=1)
    events = setup_complex_lstg(events, meta=2, leaf=1, starter=11)
    events.index = events.index.droplevel(['leaf', 'cndtn'])
    actual = get_all_cat_feats(events, levels=['meta'])
    byr_offers = [5, 3, 2, 1, 2, 2, 1, 2, 1, 0, 0]
    exp_array = [sum(byr_offers) - curr for curr in byr_offers]
    exp_array = [curr_exp / 10 for curr_exp in exp_array]
    exp = make_exp(None, exp_array)
    exp = make_exp(exp, exp_array)['exp']
    assert np.all(np.isclose(exp.values, actual['meta_byr_offers'].values))


def test_byr_offers_leaf():
    events = setup_complex_lstg(None, meta=1, leaf=2)
    events = setup_complex_lstg(events, meta=1, leaf=1, starter=11)
    events.index = events.index.droplevel(['cndtn'])
    actual = get_all_cat_feats(events, levels=['meta', 'leaf'])
    byr_offers = [5, 3, 2, 1, 2, 2, 1, 2, 1, 0, 0]
    exp_array = [sum(byr_offers) - curr for curr in byr_offers]
    exp_array = [curr_exp / 10 for curr_exp in exp_array]
    exp = make_exp(None, exp_array)
    exp = make_exp(exp, exp_array)['exp']
    assert np.all(np.isclose(exp.values, actual['leaf_byr_offers'].values))

    total = sum(byr_offers) * 2
    exp_array = [total - curr for curr in byr_offers]
    exp_array = [curr_exp / 21 for curr_exp in exp_array]
    exp = make_exp(None, exp_array)
    exp = make_exp(exp, exp_array)['exp']
    assert np.all(np.isclose(exp, actual['meta_byr_offers'].values))

    events = setup_complex_lstg(None, meta=2, leaf=1)
    events = setup_complex_lstg(events, meta=1, leaf=1, starter=11)
    events.index = events.index.droplevel(['cndtn'])
    actual = get_all_cat_feats(events, levels=['meta', 'leaf'])
    byr_offers = [5, 3, 2, 1, 2, 2, 1, 2, 1, 0, 0]
    exp_array = [sum(byr_offers) - curr for curr in byr_offers]
    exp_array = [curr_exp / 10 for curr_exp in exp_array]
    exp = make_exp(None, exp_array)
    exp = make_exp(exp, exp_array)['exp']
    assert np.all(np.isclose(exp.values, actual['leaf_byr_offers'].values))
    assert np.all(np.isclose(exp.values, actual['meta_byr_offers'].values))


def test_threads_meta():
    events = setup_complex_lstg(None, meta=1, leaf=1)
    events = setup_complex_lstg(events, meta=2, leaf=1, starter=11)
    events.index = events.index.droplevel(['leaf', 'cndtn'])
    actual = get_all_cat_feats(events, levels=['meta'])
    threads = [3, 2, 1, 1, 2, 2, 1, 1, 1, 0, 0]
    total = sum(threads)
    exp_array = [total - curr for curr in threads]
    exp_array = [curr_exp / 10 for curr_exp in exp_array]
    exp = make_exp(None, exp_array)
    exp = make_exp(exp, exp_array)['exp']
    assert np.all(np.isclose(exp.values, actual['meta_threads'].values))


def test_threads_leaf():
    events = setup_complex_lstg(None, meta=1, leaf=2)
    events = setup_complex_lstg(events, meta=1, leaf=1, starter=11)
    events.index = events.index.droplevel(['cndtn'])
    actual = get_all_cat_feats(events, levels=['meta', 'leaf'])
    threads = [3, 2, 1, 1, 2, 2, 1, 1, 1, 0, 0]
    exp_array = [sum(threads) - curr for curr in threads]
    exp_array = [curr_exp / 10 for curr_exp in exp_array]
    exp = make_exp(None, exp_array)
    exp = make_exp(exp, exp_array)['exp']
    assert np.all(np.isclose(exp.values, actual['leaf_threads'].values))

    total = sum(threads) * 2
    exp_array = [total - curr for curr in threads]
    exp_array = [curr_exp / 21 for curr_exp in exp_array]
    exp = make_exp(None, exp_array)
    exp = make_exp(exp, exp_array)['exp']
    assert np.all(np.isclose(exp, actual['meta_threads'].values))

    events = setup_complex_lstg(None, meta=2, leaf=1)
    events = setup_complex_lstg(events, meta=1, leaf=1, starter=11)
    events.index = events.index.droplevel(['cndtn'])
    actual = get_all_cat_feats(events, levels=['meta', 'leaf'])
    threads = [3, 2, 1, 1, 2, 2, 1, 1, 1, 0, 0]
    exp_array = [sum(threads) - curr for curr in threads]
    exp_array = [curr_exp / 10 for curr_exp in exp_array]
    exp = make_exp(None, exp_array)
    exp = make_exp(exp, exp_array)['exp']
    assert np.all(np.isclose(exp.values, actual['leaf_threads'].values))
    assert np.all(np.isclose(exp.values, actual['meta_threads'].values))


def test_accepts_meta():
    events = setup_complex_lstg(None, meta=1, leaf=1)
    events = setup_complex_lstg(events, meta=2, leaf=1, starter=11)
    events.index = events.index.droplevel(['leaf', 'cndtn'])
    actual = get_all_cat_feats(events, levels=['meta'])
    accepts = [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    total = sum(accepts)
    exp_array = [total - curr for curr in accepts]
    exp_array = [curr_exp / 10 for curr_exp in exp_array]
    exp = make_exp(None, exp_array)
    exp = make_exp(exp, exp_array)['exp']
    assert np.all(np.isclose(exp.values, actual['meta_accepts'].values))


def test_accepts_leaf():
    events = setup_complex_lstg(None, meta=1, leaf=2)
    events = setup_complex_lstg(events, meta=1, leaf=1, starter=11)
    events.index = events.index.droplevel(['cndtn'])
    actual = get_all_cat_feats(events, levels=['meta', 'leaf'])
    accepts = [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    exp_array = [sum(accepts) - curr for curr in accepts]
    exp_array = [curr_exp / 10 for curr_exp in exp_array]
    exp = make_exp(None, exp_array)
    exp = make_exp(exp, exp_array)['exp']
    assert np.all(np.isclose(exp.values, actual['leaf_accepts'].values))

    total = sum(accepts) * 2
    exp_array = [total - curr for curr in accepts]
    exp_array = [curr_exp / 21 for curr_exp in exp_array]
    exp = make_exp(None, exp_array)
    exp = make_exp(exp, exp_array)['exp']
    assert np.all(np.isclose(exp, actual['meta_accepts'].values))

    events = setup_complex_lstg(None, meta=2, leaf=1)
    events = setup_complex_lstg(events, meta=1, leaf=1, starter=11)
    events.index = events.index.droplevel(['cndtn'])
    actual = get_all_cat_feats(events, levels=['meta', 'leaf'])
    accepts = [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    exp_array = [sum(accepts) - curr for curr in accepts]
    exp_array = [curr_exp / 10 for curr_exp in exp_array]
    exp = make_exp(None, exp_array)
    exp = make_exp(exp, exp_array)['exp']
    assert np.all(np.isclose(exp.values, actual['leaf_accepts'].values))
    assert np.all(np.isclose(exp.values, actual['meta_accepts'].values))


def test_accept_norm_meta():
    events = setup_complex_lstg(None, meta=1, leaf=1)
    events = setup_complex_lstg(events, meta=2, leaf=1, starter=11)
    events.index = events.index.droplevel(['leaf', 'cndtn'])
    actual = get_all_cat_feats(events, levels=['meta'])
    accepts = [np.NaN, 80, 70, 50, 60, 25, 100, 95, 85, np.NaN, np.NaN]
    accepts = [accept / 100 for accept in accepts]
    accepts = np.array(accepts)
    for q in [.25, .75, 1]:
        exp_array = list()
        for i in range(len(accepts)):
            curr = np.delete(accepts, i)
            val = np.nanquantile(curr, q=q, interpolation='lower')
            if np.isnan(val):
                val = 0
            exp_array.append(val)
        exp = make_exp(None, exp_array)
        exp = make_exp(exp, exp_array)['exp']
        name = 'meta_accept_norm_{}'.format(int(q * 100))
        assert np.all(np.isclose(exp.values, actual[name].values))


def test_accept_norm_leaf():
    events = setup_complex_lstg(None, meta=1, leaf=2)
    events = setup_complex_lstg(events, meta=1, leaf=1, starter=11)
    events.index = events.index.droplevel(['cndtn'])
    actual = get_all_cat_feats(events, levels=['meta', 'leaf'])
    accepts = [np.NaN, 80, 70, 50, 60, 25, 100, 95, 85, np.NaN, np.NaN]
    accepts = [accept / 100 for accept in accepts]
    accepts = np.array(accepts)
    for q in [.25, .75, 1]:
        exp_array = list()
        for i in range(len(accepts)):
            curr = np.delete(accepts, i)
            val = np.nanquantile(curr, q=q, interpolation='lower')
            if np.isnan(val):
                val = 0
            exp_array.append(val)
        exp = make_exp(None, exp_array)
        exp = make_exp(exp, exp_array)['exp']
        name = 'leaf_accept_norm_{}'.format(int(q * 100))
        assert np.all(np.isclose(exp.values, actual[name].values))

    accepts = [np.NaN, 80, 70, 50, 60, 25, 100, 95, 85, np.NaN, np.NaN]
    accepts = accepts + accepts
    accepts = [accept / 100 for accept in accepts]
    accepts = np.array(accepts)
    for q in [.25, .75, 1]:
        exp_array = list()
        for i in range(len(accepts)):
            curr = np.delete(accepts, i)
            val = np.nanquantile(curr, q=q, interpolation='lower')
            if np.isnan(val):
                val = 0
            exp_array.append(val)
        exp = make_exp(None, exp_array)['exp']
        name = 'meta_accept_norm_{}'.format(int(q * 100))
        assert np.all(np.isclose(exp.values, actual[name].values))

    events = setup_complex_lstg(None, meta=2, leaf=1)
    events = setup_complex_lstg(events, meta=1, leaf=1, starter=11)
    events.index = events.index.droplevel(['cndtn'])
    actual = get_all_cat_feats(events, levels=['meta', 'leaf'])
    accepts = [np.NaN, 80, 70, 50, 60, 25, 100, 95, 85, np.NaN, np.NaN]
    accepts = [accept / 100 for accept in accepts]
    accepts = np.array(accepts)
    for q in [.25, .75, 1]:
        exp_array = list()
        for i in range(len(accepts)):
            curr = np.delete(accepts, i)
            val = np.nanquantile(curr, q=q, interpolation='lower')
            if np.isnan(val):
                val = 0
            exp_array.append(val)
        exp = make_exp(None, exp_array)
        exp = make_exp(exp, exp_array)['exp']
        name = '_accept_norm_{}'.format(int(q * 100))
        assert np.all(np.isclose(exp.values, actual['meta{}'.format(name)].values))
        assert np.all(np.isclose(exp.values, actual['leaf{}'.format(name)].values))


def test_con_quantile_meta():
    events = setup_complex_lstg(None, meta=1, leaf=1)
    events = setup_complex_lstg(events, meta=2, leaf=1, starter=11)
    actual = get_all_cat_feats(events, levels= ['meta'])
    offers = {
        0: [50, 30, 50],
        2: [50],
        1: [50, 55],
        3: [50],
        4: [75, 60],
        5: [20, 25],
        6: [60],
        7: [90],
        8: [30],
        9: [],
        10: []
    }
    for q in [.25, .75, 1]:
        exp_array = list()
        for i in range(len(offers)):
            considered = list()
            for j in range(len(offers)):
                if j != i:
                    considered = considered + offers[j]
            considered = np.array(considered)
            considered = considered / 100
            exp_array.append(np.nanquantile(considered, q=q, interpolation='lower'))
        exp = make_exp(None, exp_array)
        exp = make_exp(exp, exp_array)['exp']
        name = '_first_offer_{}'.format(int(q * 100))
        assert np.all(np.isclose(exp.values, actual['meta{}'.format(name)].values))


def test_con_quantile_leaf():
    events = setup_complex_lstg(None, meta=1, leaf=2)
    events = setup_complex_lstg(events, meta=1, leaf=1, starter=11)
    events.index = events.index.droplevel(['cndtn'])
    actual = get_all_cat_feats(events, levels=['meta', 'leaf'])
    offers = {
        0: [50, 30, 50],
        2: [50],
        1: [50, 55],
        3: [50],
        4: [75, 60],
        5: [20, 25],
        6: [60],
        7: [90],
        8: [30],
        9: [],
        10: []
    }
    for q in [.25, .75, 1]:
        exp_array = list()
        for i in range(len(offers)):
            considered = list()
            for j in range(len(offers)):
                if j != i:
                    considered = considered + offers[j]
            considered = np.array(considered)
            considered = considered / 100
            exp_array.append(np.nanquantile(considered, q=q, interpolation='lower'))
        exp = make_exp(None, exp_array)
        exp = make_exp(exp, exp_array)['exp']
        name = '_first_offer_{}'.format(int(q * 100))
        assert np.all(np.isclose(exp.values, actual['leaf{}'.format(name)].values))

    offers = {
        0: [50, 30, 50],
        1: [50],
        2: [50, 55],
        3: [50],
        4: [75, 60],
        5: [20, 25],
        6: [60],
        7: [90],
        8: [30],
        9: [],
        10: []
    }
    for i in range(len(offers)):
        offers[i + 11] = offers[i].copy()
    for q in [.25, .75, 1]:
        exp_array = list()
        for i in range(len(offers)):
            considered = list()
            for j in range(len(offers)):
                if j != i:
                    considered = considered + offers[j]
            considered = np.array(considered)
            considered = considered / 100
            exp_array.append(np.nanquantile(considered, q=q, interpolation='lower'))
        exp = make_exp(None, exp_array)['exp']
        name = '_first_offer_{}'.format(int(q * 100))
        assert np.all(np.isclose(exp.values, actual['meta{}'.format(name)].values))

    events = setup_complex_lstg(None, meta=2, leaf=1)
    events = setup_complex_lstg(events, meta=1, leaf=1, starter=11)
    events.index = events.index.droplevel(['cndtn'])
    actual = get_all_cat_feats(events, levels=['meta', 'leaf'])
    offers = {
        0: [50, 30, 50],
        1: [50],
        2: [50, 55],
        3: [50],
        4: [75, 60],
        5: [20, 25],
        6: [60],
        7: [90],
        8: [30],
        9: [],
        10: []
    }
    for q in [.25, .75, 1]:
        exp_array = list()
        for i in range(len(offers)):
            considered = list()
            for j in range(len(offers)):
                if j != i:
                    considered = considered + offers[j]
            considered = np.array(considered)
            considered = considered / 100
            exp_array.append(np.nanquantile(considered, q=q, interpolation='lower'))
        exp = make_exp(None, exp_array)
        exp = make_exp(exp, exp_array)['exp']
        name = '_first_offer_{}'.format(int(q * 100))
        assert np.all(np.isclose(exp.values, actual['leaf{}'.format(name)].values))
        assert np.all(np.isclose(exp.values, actual['meta{}'.format(name)].values))


def calc_bin(bin, threads):
    assert len(bin) == len(threads)
    out = list()
    bin_sum = sum(bin)
    threads_sum = sum(threads)
    for i in range(len(bin)):
        div = (threads_sum - threads[i])
        if div == 0:
            out.append(0)
        else:
            curr = (bin_sum - bin[i]) / div
            out.append(curr)
    return out


def test_bin_perc_meta():
    events = setup_complex_lstg(None, meta=1, leaf=1)

    # add a group where there are 2 lstgs and no threads
    offer = {
        'lstg': 23,
        'clock': 300
    }
    events = add_event(events, offer, trigger_type=NEW_LSTG, meta=3, leaf=1)
    offer['clock'] = 325
    events = add_event(events, offer, trigger_type=EXPIRE_LSTG, meta=3, leaf=1)

    offer = {
        'lstg': 24,
        'clock': 350
    }
    events = add_event(events, offer, trigger_type=NEW_LSTG, meta=3, leaf=1)
    offer['clock'] = 355
    events = add_event(events, offer, trigger_type=EXPIRE_LSTG, meta=3, leaf=1)

    # add a group where there are 2 lstgs, several threads, and 1 bin
    offer = {
        'lstg': 25,
        'clock': 360
    }
    events = add_event(events, offer, trigger_type=NEW_LSTG, meta=4, leaf=1)
    offer['clock'] = 365
    offer['byr'] = True
    offer['price'] = 35
    offer['thread'] = 1
    events = add_event(events, offer, trigger_type=OFFER, meta=4, leaf=1)

    offer['price'] = 45
    offer['clock'] = 370
    offer['thread'] = 2
    events = add_event(events, offer, trigger_type=OFFER, meta=4, leaf=1)

    offer['price'] = 55
    offer['clock'] = 375
    offer['thread'] = 3
    events = add_event(events, offer, trigger_type=OFFER, meta=4, leaf=1)

    offer['price'] = 60
    offer['clock'] = 380
    offer['thread'] = 4
    events = add_event(events, offer, trigger_type=OFFER, meta=4, leaf=1)

    offer['clock'] = 385
    offer['lstg'] = 26
    events = add_event(events, offer, trigger_type=NEW_LSTG, meta=4, leaf=1)

    offer['thread'] = 1
    offer['price'] = 50
    offer['clock'] = 387
    offer['byr'] = True
    events = add_event(events, offer, trigger_type=OFFER, meta=4, leaf=1)

    offer['thread'] = 2
    offer['price'] = 100
    offer['clock'] = 389
    events = add_event(events, offer, trigger_type=ACCEPTANCE, meta=4, leaf=1)
    offer['thread'] = 1
    offer['byr'] = False
    events = add_event(events, offer, trigger_type=CENSORED, meta=4, leaf=1)

    offer['clock'] = 400
    offer['byr'] = False
    offer['lstg'] = 25
    offer['price'] = 60
    offer['thread'] = 4
    events = add_event(events, offer, trigger_type=ACCEPTANCE, meta=4, leaf=1)
    offer['thread'] = 3
    events = add_event(events, offer, trigger_type=CENSORED, meta=4, leaf=1)
    offer['thread'] = 2
    events = add_event(events, offer, trigger_type=CENSORED, meta=4, leaf=1)
    offer['thread'] = 1
    events = add_event(events, offer, trigger_type=CENSORED, meta=4, leaf=1)

    threads = [1] * 11
    bin = [0] * 11
    threads2 = [1, 1]
    bin2 = [0] * 2
    threads3 = [1, 1]
    bin3 = [0, 1]

    offers1 = {
        0: [50, 30, 50],
        2: [50],
        1: [50, 55],
        3: [50],
        4: [75, 60],
        5: [20, 25],
        6: [60],
        7: [90],
        8: [30],
        9: [],
        10: []
    }
    offers2 = {
        0: [],
        1: []
    }
    offers3 = {
        0: [35, 45, 55, 60],
        1: [50]
    }
    events.index = events.index.droplevel(['leaf', 'cndtn'])

    actual = get_all_cat_feats(events, levels=['meta'])
    perc_exp1 = calc_bin(bin, threads)
    perc_exp2 = calc_bin(bin2, threads2)
    perc_exp3 = calc_bin(bin3, threads3)
    perc_exp = np.concatenate([perc_exp1, perc_exp2, perc_exp3])
    print('')
    print('perc_exp')
    print(perc_exp)
    print(actual['meta_bin'].values)
    assert np.all(np.isclose(perc_exp, actual['meta_bin'].values))
    for q in [.25, .75, 1]:
        exp_array = list()
        for offers in [offers1, offers2, offers3]:
            for i in range(len(offers)):
                considered = list()
                for j in range(len(offers)):
                    if j != i:
                        considered = considered + offers[j]
                considered = np.array(considered)
                considered = considered / 100
                quant = np.nanquantile(considered, q=q, interpolation='lower')
                if np.isnan(quant):
                    quant = 0
                exp_array.append(quant)
        exp = make_exp(None, exp_array)['exp']
        name = '_first_offer_{}'.format(int(q * 100))
        assert np.all(np.isclose(exp.values, actual['meta{}'.format(name)].values))

    # check accept norm
    accepts1 = [np.NaN, .80, .70, .50, .60, .25, 1.00, .95, .85, np.NaN, np.NaN]
    accepts2 = [np.NaN, np.NaN]
    accepts3 = [.6, np.NaN]
    actual = get_all_cat_feats(events, levels=['meta'])
    for q in [.25, .75, 1]:
        exp_array = list()
        for offers in [accepts1, accepts2, accepts3]:
            for i in range(len(offers)):
                considered = list()
                for j in range(len(offers)):
                    if j != i:
                        considered.append(offers[j])
                considered = np.array(considered)
                quant = np.nanquantile(considered, q=q, interpolation='lower')
                if np.isnan(quant):
                    quant = 0
                exp_array.append(quant)
        exp = make_exp(None, exp_array)['exp']
        name = '_accept_norm_{}'.format(int(q * 100))
        assert np.all(np.isclose(exp.values, actual['meta{}'.format(name)].values))


def test_bin_perc_leaf():
    events = setup_complex_lstg(None, meta=1, leaf=1)

    # add a group where there are 2 lstgs and no threads
    offer = {
        'lstg': 23,
        'clock': 300
    }
    events = add_event(events, offer, trigger_type=NEW_LSTG, meta=1, leaf=2)
    offer['clock'] = 325
    events = add_event(events, offer, trigger_type=EXPIRE_LSTG, meta=1, leaf=2)

    offer = {
        'lstg': 24,
        'clock': 350
    }
    events = add_event(events, offer, trigger_type=NEW_LSTG, meta=1, leaf=2)
    offer['clock'] = 355
    events = add_event(events, offer, trigger_type=EXPIRE_LSTG, meta=1, leaf=2)

    # add a group where there are 2 lstgs, several threads, and 1 bin
    offer = {
        'lstg': 25,
        'clock': 360
    }
    events = add_event(events, offer, trigger_type=NEW_LSTG, meta=1, leaf=3)
    offer['clock'] = 365
    offer['byr'] = True
    offer['price'] = 35
    offer['thread'] = 1
    events = add_event(events, offer, trigger_type=OFFER, meta=1, leaf=3)

    offer['price'] = 45
    offer['clock'] = 370
    offer['thread'] = 2
    events = add_event(events, offer, trigger_type=OFFER, meta=1, leaf=3)

    offer['price'] = 55
    offer['clock'] = 375
    offer['thread'] = 3
    events = add_event(events, offer, trigger_type=OFFER, meta=1, leaf=3)

    offer['price'] = 60
    offer['clock'] = 380
    offer['thread'] = 4
    events = add_event(events, offer, trigger_type=OFFER, meta=1, leaf=3)

    offer['clock'] = 385
    offer['lstg'] = 26
    events = add_event(events, offer, trigger_type=NEW_LSTG, meta=1, leaf=3)

    offer['thread'] = 1
    offer['price'] = 50
    offer['clock'] = 387
    offer['byr'] = True
    events = add_event(events, offer, trigger_type=OFFER, meta=1, leaf=3)

    offer['thread'] = 2
    offer['price'] = 100
    offer['clock'] = 389
    events = add_event(events, offer, trigger_type=ACCEPTANCE,  meta=1, leaf=3)
    offer['thread'] = 1
    offer['byr'] = False
    events = add_event(events, offer, trigger_type=CENSORED,  meta=1, leaf=3)

    offer['clock'] = 400
    offer['byr'] = False
    offer['lstg'] = 25
    offer['price'] = 60
    offer['thread'] = 4
    events = add_event(events, offer, trigger_type=ACCEPTANCE,  meta=1, leaf=3)
    offer['thread'] = 3
    events = add_event(events, offer, trigger_type=CENSORED,  meta=1, leaf=3)
    offer['thread'] = 2
    events = add_event(events, offer, trigger_type=CENSORED,  meta=1, leaf=3)
    offer['thread'] = 1
    events = add_event(events, offer, trigger_type=CENSORED,  meta=1, leaf=3)

    events.byr = events.byr.astype(bool)
    events.reject = events.reject.astype(bool)
    events.accept = events.accept.astype(bool)
    events.censored = events.censored.astype(bool)
    events.flag = events.flag.astype(bool)
    events.price = events.price.astype(np.int64)
    events.clock = events.clock.astype(np.int64)

    threads = [1] * 11
    bin = [0] * 11
    threads2 = [1, 1]
    bin2 = [0] * 2
    threads3 = [1] * 2
    bin3 = [0, 1]

    offers1 = {
        0: [50, 30, 50],
        1: [50],
        2: [50, 55],
        3: [50],
        4: [75, 60],
        5: [20, 25],
        6: [60],
        7: [90],
        8: [30],
        9: [],
        10: []
    }
    offers2 = {
        0: [],
        1: []
    }
    offers3 = {
        0: [35, 45, 55, 60],
        1: [50]
    }

    events.index = events.index.droplevel(['cndtn'])

    actual = get_all_cat_feats(events, levels=['meta', 'leaf'])
    perc_exp1 = calc_bin(bin, threads)
    perc_exp2 = calc_bin(bin2, threads2)
    perc_exp3 = calc_bin(bin3, threads3)
    perc_exp = np.concatenate([perc_exp1, perc_exp2, perc_exp3])
    assert np.all(np.isclose(perc_exp, actual['leaf_bin'].values))
    for q in [.25, .75, 1]:
        exp_array = list()
        for offers in [offers1, offers2, offers3]:
            for i in range(len(offers)):
                considered = list()
                for j in range(len(offers)):
                    if j != i:
                        considered = considered + offers[j]
                considered = np.array(considered)
                considered = considered / 100
                quant = np.nanquantile(considered, q=q, interpolation='lower')
                if np.isnan(quant):
                    quant = 0
                exp_array.append(quant)
        exp = make_exp(None, exp_array)['exp']
        name = '_first_offer_{}'.format(int(q * 100))
        assert np.all(np.isclose(exp.values, actual['leaf{}'.format(name)].values))

    # check accept norm
    accepts1 = [np.NaN, .80, .70, .50, .60, .25, 1.00, .95, .85, np.NaN, np.NaN]
    accepts2 = [np.NaN, np.NaN]
    accepts3 = [.6, np.NaN]
    actual = get_all_cat_feats(events, levels=['meta', 'leaf'])
    for q in [.25, .75, 1]:
        exp_array = list()
        for offers in [accepts1, accepts2, accepts3]:
            for i in range(len(offers)):
                considered = list()
                for j in range(len(offers)):
                    if j != i:
                        considered.append(offers[j])
                considered = np.array(considered)
                quant = np.nanquantile(considered, q=q, interpolation='lower')
                if np.isnan(quant):
                    quant = 0
                exp_array.append(quant)
        exp = make_exp(None, exp_array)['exp']
        name = '_accept_norm_{}'.format(int(q * 100))
        assert np.all(np.isclose(exp.values, actual['leaf{}'.format(name)].values))

    # meta
    threads = [1] * 15
    bin = ([0] * 11) + [0, 0, 0, 1]

    offers = {
        0: [50, 30, 50],
        1: [50, 55],
        2: [50],
        3: [50],
        4: [75, 60],
        5: [20, 25],
        6: [60],
        7: [90],
        8: [30],
        9: [],
        10: [],
        11: [],
        12: [],
        13: [35, 45, 55, 60],
        14: [50]
    }
    actual = get_all_cat_feats(events, levels=['meta', 'leaf'])

    perc_exp = np.array(calc_bin(bin, threads))
    print('exp')
    print(perc_exp)
    print('actual')
    print(actual['meta_bin'])
    assert np.all(np.isclose(perc_exp, actual['meta_bin'].values))

    for q in [.25, .75, 1]:
        print('q: {}'.format(q))
        exp_array = list()
        for i in range(len(offers)):
            considered = list()
            for j in range(len(offers)):
                if j != i:
                    considered = considered + offers[j]
            considered = np.array(considered)
            considered = considered / 100
            quant = np.nanquantile(considered, q=q, interpolation='lower')
            if np.isnan(quant):
                quant = 0
            if i == 1 or i == 2:
                print('considered')
                print(considered)
            exp_array.append(quant)
        exp = make_exp(None, exp_array)['exp']
        name = '_first_offer_{}'.format(int(q * 100))
        print('actual')
        print(actual)
        assert np.all(np.isclose(exp.values, actual['meta{}'.format(name)].values))

    # accept norm
    offers = [np.NaN, .80, .70, .50, .60, .25, 1.00,
               .95, .85, np.NaN, np.NaN, np.NaN, np.NaN, .6, np.NaN]
    actual = get_all_cat_feats(events, levels=['meta', 'leaf'])
    for q in [.25, .75, 1]:
        exp_array = list()
        for i in range(len(offers)):
            considered = list()
            for j in range(len(offers)):
                if j != i:
                    considered.append(offers[j])
            considered = np.array(considered)
            quant = np.nanquantile(considered, q=q, interpolation='lower')
            if np.isnan(quant):
                quant = 0
            exp_array.append(quant)
        exp = make_exp(None, exp_array)['exp']
        name = '_accept_norm_{}'.format(int(q * 100))
        assert np.all(np.isclose(exp.values, actual['meta{}'.format(name)].values))


def delay_appendix_1(events, meta=1, leaf=1):
    # buyer delays up to turn 7
    # all seller offers are auto rejects
    offer = {
        'lstg': 23,
        'clock': 300
    }
    events = add_event(events, offer, trigger_type=NEW_LSTG, meta=meta, leaf=leaf)
    offer['byr'] = True
    offer['clock'] = 305
    offer['price'] = 30
    offer['thread'] = 1
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)
    offer['byr'] = False
    offer['price'] = 100
    events = add_event(events, offer, trigger_type=SLR_REJECTION, meta=meta, leaf=leaf)

    offer['byr'] = True
    offer['clock'] = 320
    offer['price'] = 40
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)
    offer['byr'] = False
    offer['price'] = 100
    events = add_event(events, offer, trigger_type=SLR_REJECTION, meta=meta, leaf=leaf)

    offer['byr'] = True
    offer['clock'] = 370
    offer['price'] = 50
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)
    offer['byr'] = False
    offer['price'] = 100
    events = add_event(events, offer, trigger_type=SLR_REJECTION, meta=meta, leaf=leaf)

    offer['byr'] = True
    offer['clock'] = 600
    offer['price'] = 50
    events = add_event(events, offer, trigger_type=BYR_REJECTION, meta=meta, leaf=leaf)

    offer = {
        'lstg': 24,
        'clock': 450
    }
    events = add_event(events, offer, trigger_type=NEW_LSTG, meta=meta, leaf=leaf)
    offer['clock'] = 500
    events = add_event(events, offer, trigger_type=EXPIRE_LSTG, meta=meta, leaf=leaf)
    return events


def delay_appendix_2(events, meta=1, leaf=1):
    # one max seller delay
    # one max buyer delay
    time = 700
    offer = {
        'lstg': 25,
        'clock': 700
    }
    events = add_event(events, offer, trigger_type=NEW_LSTG, meta=meta, leaf=leaf)
    time += 5
    offer['byr'] = True
    offer['clock'] = time
    offer['price'] = 30
    offer['thread'] = 1
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    time += 800
    offer['clock'] = time
    offer['byr'] = False
    offer['price'] = 70
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    time += 600
    offer['byr'] = True
    offer['clock'] = time
    offer['price'] = 50
    offer['thread'] = 1
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    time += MAX_DELAY['slr']
    offer['clock'] = time
    offer['byr'] = False
    offer['price'] = 30
    events = add_event(events, offer, trigger_type=SLR_REJECTION, meta=meta, leaf=leaf)

    time += MAX_DELAY['byr']
    offer['clock'] = time
    offer['byr'] = True
    offer['price'] = 50
    events = add_event(events, offer, trigger_type=BYR_REJECTION, meta=meta, leaf=leaf)

    time += 15
    offer['clock'] = time
    events = add_event(events, offer, trigger_type=EXPIRE_LSTG, meta=meta, leaf=leaf)

    time += 15
    offer = {
        'lstg': 26,
        'clock': time
    }
    events = add_event(events, offer, trigger_type=NEW_LSTG, meta=meta, leaf=leaf)
    time += 15
    offer['clock'] = time
    events = add_event(events, offer, trigger_type=EXPIRE_LSTG, meta=meta, leaf=leaf)
    return events


def test_delay_meta():
    events = setup_complex_lstg(None, meta=1, leaf=1)
    events = delay_appendix_1(events, meta=2, leaf=1)
    events = delay_appendix_2(events, meta=3, leaf=1)

    events.byr = events.byr.astype(bool)
    events.reject = events.reject.astype(bool)
    events.accept = events.accept.astype(bool)
    events.censored = events.censored.astype(bool)
    events.flag = events.flag.astype(bool)
    events.price = events.price.astype(np.int64)
    events.clock = events.clock.astype(np.int64)
    events.index = events.index.droplevel(['cndtn', 'leaf'])

    # leaf
    byr_delays = {
        0: [10, 5, 30, 15],
        1: [5, 5],
        2: [20],
        3: [],
        4: [],
        5: [20],
        6: [5],
        7: [5],
        8: [7],
        9: [],
        10: []
    }

    byr_delays2 = {
        0: [15, 50],
        1: []
    }

    byr_delays3 = {
        0: [600],
        1: []
    }

    slr_delays = {
        0: [10, 30, 10],
        1: [20, 25],
        2: [20, 10],
        3: [10],
        4: [30],
        5: [5],
        6: [15],
        7: [20],
        8: [1],
        9: [],
        10: []
    }

    slr_delays2 = {
        0: [],
        1: []
    }

    slr_delays3 = {
        0: [800],
        1: []
    }

    byr_expire = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    byr_count = [4, 2, 1, 0, 0, 1, 1, 1, 1, 0, 0]
    slr_expire = byr_expire.copy()
    slr_count = [3, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0]
    byr_expire2 = [0, 0]
    byr_count2 = [2, 0]
    slr_expire2 = byr_expire2.copy()
    slr_count2 = [0, 0]
    byr_expire3 = [1, 0]
    byr_count3 = [2, 0]
    slr_expire3 = byr_expire3.copy()
    slr_count3 = [2, 0]

    actual = get_all_cat_feats(events, levels=['meta'])

    perc_exp1 = calc_bin(byr_expire, byr_count)
    perc_exp2 = calc_bin(byr_expire2, byr_count2)
    perc_exp3 = calc_bin(byr_expire3, byr_count3)
    perc_exp = np.concatenate([perc_exp1, perc_exp2, perc_exp3])
    assert np.all(np.isclose(perc_exp, actual['meta_byr_expire'].values))

    perc_exp1 = calc_bin(slr_expire, slr_count)
    perc_exp2 = calc_bin(slr_expire2, slr_count2)
    perc_exp3 = calc_bin(slr_expire3, slr_count3)
    perc_exp = np.concatenate([perc_exp1, perc_exp2, perc_exp3])
    assert np.all(np.isclose(perc_exp, actual['meta_slr_expire'].values))

    for delay_type in ['byr', 'slr']:
        print('type: {}'.format(delay_type))
        if delay_type == 'byr':
            delay_list = [byr_delays, byr_delays2, byr_delays3]
        else:
            delay_list = [slr_delays, slr_delays2, slr_delays3]
        for q in [.25, .75, 1]:
            print('q: {}'.format(q))
            exp_array = list()
            for offers in delay_list:
                for i in range(len(offers)):
                    considered = list()
                    for j in range(len(offers)):
                        if j != i:
                            considered = considered + offers[j]
                    considered = np.array(considered)
                    considered = considered / MAX_DELAY[delay_type]
                    quant = np.nanquantile(considered, q=q, interpolation='lower')
                    if np.isnan(quant):
                        quant = 0
                    exp_array.append(quant)
            exp = make_exp(None, exp_array)['exp']
            name = '_{}_delay_{}'.format(delay_type, int(q * 100))
            assert np.all(np.isclose(exp.values, actual['meta{}'.format(name)].values))


def test_delay_leaf():
    events = setup_complex_lstg(None, meta=1, leaf=1)
    events = delay_appendix_1(events, meta=1, leaf=2)
    events = delay_appendix_2(events, meta=1, leaf=3)

    events.byr = events.byr.astype(bool)
    events.reject = events.reject.astype(bool)
    events.accept = events.accept.astype(bool)
    events.censored = events.censored.astype(bool)
    events.flag = events.flag.astype(bool)
    events.price = events.price.astype(np.int64)
    events.clock = events.clock.astype(np.int64)
    events.index = events.index.droplevel(['cndtn'])

    byr_delays_all = {
        0: [10, 5, 30, 15],
        1: [5, 5],
        2: [20],
        3: [],
        4: [],
        5: [20],
        6: [5],
        7: [5],
        8: [7],
        9: [],
        10: [],
        11: [15, 50],
        12: [],
        13: [600],
        14: []
    }
    byr_delays = {
        0: [10, 5, 30, 15],
        1: [5, 5],
        2: [20],
        3: [],
        4: [],
        5: [20],
        6: [5],
        7: [5],
        8: [7],
        9: [],
        10: []
    }

    byr_delays2 = {
        0: [15, 50],
        1: []
    }

    byr_delays3 = {
        0: [600],
        1: []
    }

    slr_delays_all = {
        0: [10, 30, 10],
        1: [20, 25],
        2: [20, 10],
        3: [10],
        4: [30],
        5: [5],
        6: [15],
        7: [20],
        8: [1],
        9: [],
        10: [],
        11: [],
        12: [],
        13: [800],
        14: []
    }

    slr_delays = {
        0: [10, 30, 10],
        1: [20, 25],
        2: [20, 10],
        3: [10],
        4: [30],
        5: [5],
        6: [15],
        7: [20],
        8: [1],
        9: [],
        10: []
    }

    slr_delays2 = {
        0: [],
        1: []
    }

    slr_delays3 = {
        0: [800],
        1: []
    }

    byr_expire = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    byr_count = [4, 2, 1, 0, 0, 1, 1, 1, 1, 0, 0]
    slr_expire = byr_expire.copy()
    slr_count = [3, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0]
    byr_expire2 = [0, 0]
    byr_count2 = [2, 0]
    slr_expire2 = byr_expire2.copy()
    slr_count2 = [0, 0]
    byr_expire3 = [1, 0]
    byr_count3 = [2, 0]
    slr_expire3 = byr_expire3.copy()
    slr_count3 = [2, 0]

    byr_expire_all = byr_expire + byr_expire2 + byr_expire3
    slr_expire_all = slr_expire + slr_expire2 + slr_expire3
    byr_count_all = byr_count + byr_count2 + byr_count3
    slr_count_all = slr_count + slr_count2 + slr_count3

    actual = get_all_cat_feats(events, levels=['meta', 'leaf'])

    perc_exp1 = calc_bin(byr_expire, byr_count)
    perc_exp2 = calc_bin(byr_expire2, byr_count2)
    perc_exp3 = calc_bin(byr_expire3, byr_count3)
    perc_exp = np.concatenate([perc_exp1, perc_exp2, perc_exp3])
    assert np.all(np.isclose(perc_exp, actual['leaf_byr_expire'].values))

    print('expected')
    print(perc_exp)


    perc_exp = np.array(calc_bin(byr_expire_all, byr_count_all))
    assert np.all(np.isclose(perc_exp, actual['meta_byr_expire'].values))


    perc_exp1 = calc_bin(slr_expire, slr_count)
    perc_exp2 = calc_bin(slr_expire2, slr_count2)
    perc_exp3 = calc_bin(slr_expire3, slr_count3)
    perc_exp = np.concatenate([perc_exp1, perc_exp2, perc_exp3])
    assert np.all(np.isclose(perc_exp, actual['leaf_slr_expire'].values))

    perc_exp = np.array(calc_bin(slr_expire_all, slr_count_all))
    assert np.all(np.isclose(perc_exp, actual['meta_slr_expire'].values))

    for delay_type in ['byr', 'slr']:
        print('type: {}'.format(delay_type))
        if delay_type == 'byr':
            delay_list = [byr_delays, byr_delays2, byr_delays3]
        else:
            delay_list = [slr_delays, slr_delays2, slr_delays3]
        for q in [.25, .75, 1]:
            print('q: {}'.format(q))
            exp_array = list()
            for offers in delay_list:
                for i in range(len(offers)):
                    considered = list()
                    for j in range(len(offers)):
                        if j != i:
                            considered = considered + offers[j]
                    considered = np.array(considered)
                    considered = considered / MAX_DELAY[delay_type]
                    quant = np.nanquantile(considered, q=q, interpolation='lower')
                    if np.isnan(quant):
                        quant = 0
                    exp_array.append(quant)
            exp = make_exp(None, exp_array)['exp']
            name = '_{}_delay_{}'.format(delay_type, int(q * 100))
            assert np.all(np.isclose(exp.values, actual['leaf{}'.format(name)].values))

    for delay_type in ['byr', 'slr']:
        print('type: {}'.format(delay_type))
        if delay_type == 'byr':
            offers = byr_delays_all
        else:
            offers = slr_delays_all
        for q in [.25, .75, 1]:
            print('q: {}'.format(q))
            exp_array = list()
            for i in range(len(offers)):
                considered = list()
                for j in range(len(offers)):
                    if j != i:
                        considered = considered + offers[j]
                considered = np.array(considered)
                considered = considered / MAX_DELAY[delay_type]
                quant = np.nanquantile(considered, q=q, interpolation='lower')
                if np.isnan(quant):
                    quant = 0
                exp_array.append(quant)
            exp = make_exp(None, exp_array)['exp']
            name = '_{}_delay_{}'.format(delay_type, int(q * 100))
            assert np.all(np.isclose(exp.values, actual['meta{}'.format(name)].values))


def test_start_price_pctile_meta():
    prices = [.8, .7, .9, .4, .6, .75, .2, .3, .5, .65, .1]
    check_quantiles_wrapper(prices, leaf=False, feat_ind=5)


def test_start_price_pctile_leaf():
    prices = [.8, .7, .9, .4, .6, .75, .2, .3, .5, .65, .1]
    check_quantiles_wrapper(prices, leaf=True, feat_ind=5)


def test_arrival_rate_pctile_meta():
    rates = [.4, .2, .3, .6, .65, .7, .45, .35, .55, .8, .9]
    check_quantiles_wrapper(rates, leaf=False, feat_ind=7)


def test_arrival_rate_pctile_leaf():
    rates = [.4, .2, .3, .6, .65, .7, .45, .35, .55, .8, .9]
    check_quantiles_wrapper(rates, leaf=True, feat_ind=7)


def test_byr_hist_pctile_meta():
    byr_hist = {
        0: [.2, .9, .6],
        1: [.7, .25],
        2: [.15],
        3: [.75],
        4: [.5, .65],
        5: [.25, .3],
        6: [.8],
        7: [.85],
        8: [.95],
        9: [],
        10: []
    }
    check_quantiles_wrapper(byr_hist, leaf=False, feat_ind=6)


def test_byr_hist_pctile_leaf():
    byr_hist = {
        0: [.2, .9, .6],
        1: [.7, .25],
        2: [.15],
        3: [.75],
        4: [.5, .65],
        5: [.25, .3],
        6: [.8],
        7: [.85],
        8: [.95],
        9: [],
        10: []
    }
    check_quantiles_wrapper(byr_hist, leaf=True, feat_ind=6)


def check_quantiles_wrapper(offers, leaf=True, feat_ind=0):
    if feat_ind == 5:
        featname = 'start_price_pctile'
    elif feat_ind == 6:
        featname = 'byr_hist'
    elif feat_ind == 7:
        featname = 'arrival_rate'
    else:
        raise NotImplementedError()
    if leaf:
        events = setup_complex_lstg(None, meta=1, leaf=1)
        events = setup_complex_lstg(events, meta=1, leaf=2, starter=11)
        events.index = events.index.droplevel(['cndtn'])
        actual = get_all_cat_feats(events, levels=['meta', 'leaf'])
        check_quantiles(offers, actual, level_type='leaf', featname=featname, double=True)
        if isinstance(offers, dict):
            prev = len(offers)
            for i in range(prev):
                offers[i + prev] = offers[i].copy()
            meta_offers = offers
        else:
            meta_offers = offers + offers
        check_quantiles(meta_offers, actual, level_type='meta', featname=featname, double=False)
    else:
        events = setup_complex_lstg(None, meta=1, leaf=1)
        events = setup_complex_lstg(events, meta=2, leaf=1, starter=11)
        events.index = events.index.droplevel(['leaf', 'cndtn'])
        actual = get_all_cat_feats(events, levels=['meta'])
        check_quantiles(offers, actual, level_type='meta', featname=featname, double=True)


def check_quantiles(offers, actual, level_type='leaf', featname=None, double=False):
    for q in [.25, .75, 1]:
        print('q: {}'.format(q))
        exp_array = list()
        for i in range(len(offers)):
            considered = list()
            for j in range(len(offers)):
                if j != i:
                    if isinstance(offers, dict):
                        considered = considered + offers[j]
                    else:
                        considered.append(offers[j])
            considered = np.array(considered)
            quant = np.nanquantile(considered, q=q, interpolation='lower')
            if np.isnan(quant):
                quant = 0
            exp_array.append(quant)
        exp = make_exp(None, exp_array)
        if double:
            exp = make_exp(exp, exp_array)
        exp = exp['exp']
        print('exp')
        print(exp)
        name = '{}_{}_{}'.format(level_type, featname, int(q * 100))
        assert np.all(np.isclose(exp.values, actual[name].values))


def add_empty_lstg(events, meta=1, leaf=1, starter=11):
    offer = {
        'lstg': 1 + starter,
        'clock': 300
    }
    events = add_event(events, offer=offer, trigger_type=NEW_LSTG, meta=meta, leaf=leaf)

    offer['clock'] = 450
    events = add_event(events, offer=offer, trigger_type=EXPIRE_LSTG, meta=meta, leaf=leaf)
    return events


def add_active_lstg(events, meta=1, leaf=1, starter=11):
    # start new lstg = 1
    offer = {
        'lstg': 1 + starter,
        'clock': 15,
        'start_price_pctile': .8,
        'arrival_rate': .4
    }
    events = add_event(events, offer, trigger_type=NEW_LSTG, meta=meta, leaf=leaf)

    # start new thread =1 for lstg = 1
    offer['price'] = 50
    offer['thread'] = 1
    offer['byr'] = True
    offer['byr_hist'] = .2
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
    offer['byr_hist'] = .9
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # slr auto reject for thread = 2
    offer['byr'] = False
    offer['price'] = 100
    events = add_event(events, offer, trigger_type=SLR_REJECTION, meta=meta, leaf=leaf)

    # byr offer in thread = 2 for lstg = 1
    offer['thread'] = 2
    offer['clock'] = 110
    offer['price'] = 70
    events = add_event(events, offer, trigger_type=OFFER, meta=meta, leaf=leaf)

    # slr counter offer in thread = 2 for lstg = 1
    offer['byr'] = False
    offer['price'] = 75
    offer['thread'] = 2
    offer['lstg'] = 1 + starter
    offer['clock'] = 120
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
    return events

def test_one_lstg_no_arrivals_meta():
    events = setup_complex_lstg(None, meta=1, leaf=1)
    events = add_empty_lstg(events, meta=2, leaf=1, starter=11)
    events.index = events.index.droplevel(['leaf', 'cndtn'])

    actual = get_all_cat_feats(events, levels=['meta'])
    actual = actual.loc[12, :].values
    actual = np.squeeze(actual)
    exp = np.zeros(actual.shape)
    assert np.all(np.isclose(exp, actual))


def test_one_lstg_active_meta():
    events = setup_complex_lstg(None, meta=1, leaf=1)
    events = add_active_lstg(events, meta=2, leaf=1, starter=11)
    events.index = events.index.droplevel(['leaf', 'cndtn'])

    actual = get_all_cat_feats(events, levels=['meta'])
    actual = actual.loc[12, :].values
    actual = np.squeeze(actual)
    exp = np.zeros(actual.shape)
    assert np.all(np.isclose(exp, actual))



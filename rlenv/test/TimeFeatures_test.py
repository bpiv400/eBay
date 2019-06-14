from TimeFeatures import TimeFeatures
import pytest
from pprint import pprint
import time_triggers

@pytest.fixture
def lstgs():
    lstg1 = {
        'slr': 1,
        'meta': 2,
        'leaf': 3,
        'title': 4,
        'cndtn': 5,
        'lstg': 6,
        'byr_count': 0,
        'slr_count': 0,
        'thread_id': 10
    }

    lstg2 = {
        'slr': 1,
        'meta': 2,
        'leaf': 3,
        'title': 4,
        'cndtn': 5,
        'lstg': 7,
        'byr_count': 0,
        'slr_count': 0,
        'thread_id': 11
    }

    lstg3 = {
        'slr': 1,
        'meta': 2,
        'leaf': 3,
        'title': 4,
        'cndtn': 6,
        'lstg': 8,
        'byr_count': 0,
        'slr_count': 0,
        'thread_id': 12
    }
    return lstg1, lstg2, lstg3

@pytest.fixture
def timefeats():
    return TimeFeatures(expiration=15)

@pytest.fixture
def init_timefeats():
    lstg1 = {
        'slr': 1,
        'meta': 2,
        'leaf': 3,
        'title': 4,
        'cndtn': 5,
        'lstg': 6,
        'byr_count': 0,
        'slr_count': 0,
        'thread_id': 10
    }

    lstg2 = {
        'slr': 1,
        'meta': 2,
        'leaf': 3,
        'title': 4,
        'cndtn': 5,
        'lstg': 7,
        'byr_count': 0,
        'slr_count': 0,
        'thread_id': 11
    }

    lstg3 = {
        'slr': 1,
        'meta': 2,
        'leaf': 3,
        'title': 4,
        'cndtn': 6,
        'lstg': 8,
        'byr_count': 0,
        'slr_count': 0,
        'thread_id': 12
    }
    timefeats = TimeFeatures(expiration=15)
    timefeats.initialize_time_feats(lstg1, start_price=5)
    timefeats.initialize_time_feats(lstg2, start_price=100)
    timefeats.initialize_time_feats(lstg3, start_price=150)
    return timefeats


def test_initialize_feats(lstgs, timefeats):
    lstg1, lstg2, lstg3 = lstgs
    timefeats.initialize_time_feats(lstg1, start_price=5)
    timefeats.initialize_time_feats(lstg2, start_price=100)
    timefeats.initialize_time_feats(lstg3, start_price=150)
    assert timefeats.lstg_active(lstg1)
    assert timefeats.lstg_active(lstg2)
    assert timefeats.lstg_active(lstg3)
    # pprint(timefeats.get_feats(lstg1, time=15))
    # pprint(timefeats.get_feats(lstg3, time=15))


def test_byr_offer(lstgs, init_timefeats):
    lstg1, lstg2, lstg3 = lstgs
    timefeats = init_timefeats
    prev = timefeats.get_feats(lstg1, time=2)
    timefeats.update_features(trigger_type=time_triggers.BUYER_OFFER, ids=lstg1,
                              offer={
                                  'byr': True,
                                  'time': 5,
                                  'price': 3
                              })
    next = timefeats.get_feats(lstg1, time=5)
    for feat in prev:
        if 'byr_offers' in feat:
            assert next[feat] == 0
        elif 'slr_offers' in feat:
            assert next[feat] == 0
        elif 'open_threads' in feat:
            assert next[feat] == 0
        elif 'open_lstgs' in feat:
            if 'cndtn' not in feat:
                assert next[feat] == 3
            else:
                assert next[feat] == 2
        elif 'byr_max' in feat:
            assert next[feat] == 0
        elif 'slr_min' in feat:
            if 'lstg' in feat:
                assert next[feat] == 5
            else:
                assert next[feat] == 5
    del lstg2['thread_id']
    next = timefeats.get_feats(lstg2, time=6)
    for feat in prev:
        if 'byr_offers' in feat:
            if 'lstg' in feat:
                assert next[feat] == 0
            else:
                assert next[feat] == 1
        elif 'slr_offers' in feat:
            assert next[feat] == 0
        elif 'open_threads' in feat:
            if 'lstg' in feat:
                assert next[feat] == 0
            else:
                assert next[feat] == 1
        elif 'open_lstgs' in feat:
            if 'cndtn' not in feat:
                assert next[feat] == 3
            else:
                assert next[feat] == 2
        elif 'byr_max' in feat:
            if 'lstg' not in feat:
                assert next[feat] == 3
            else:
                assert next[feat] == 0
        elif 'slr_min' in feat:
            if 'lstg' in feat:
                assert next[feat] == 100
            else:
                assert next[feat] == 5

    timefeats.update_features(trigger_type=time_triggers.BUYER_OFFER, ids=lstg3,
                              offer={
                                  'byr': True,
                                  'time': 6,
                                  'price': 3
                              })
    next = timefeats.get_feats(lstg3, 6)
    for feat in prev:
        if 'byr_offers' in feat:
            if 'lstg' in feat or 'cndtn' in feat:
                assert next[feat] == 0
            else:
                assert next[feat] == 1
        elif 'slr_offers' in feat:
            assert next[feat] == 0
        elif 'open_threads' in feat:
            if 'lstg' in feat or 'cndtn' in feat:
                assert next[feat] == 0
            else:
                assert next[feat] == 1
        elif 'open_lstgs' in feat:
            if 'cndtn' not in feat:
                assert next[feat] == 3
            else:
                assert next[feat] == 1
        elif 'byr_max' in feat:
            if 'lstg' not in feat and 'cndtn' not in feat:
                assert next[feat] == 3
            else:
                assert next[feat] == 0
        elif 'slr_min' in feat:
            if 'lstg' in feat or 'cndtn' in feat:
                assert next[feat] == 150
            else:
                assert next[feat] == 5

def test_multiple_threads_offers(init_timefeats, lstgs):
    lstg1, lstg2, lstg3 = lstgs
    timefeats = init_timefeats
    lstg4 = lstg1.copy()
    lstg4['thread_id'] = 15
    timefeats.update_features(trigger_type=time_triggers.BUYER_OFFER, ids=lstg1,
                              offer={
                                  'byr': True,
                                  'time': 1,
                                  'price': 2
                              })
    timefeats.update_features(trigger_type=time_triggers.BUYER_OFFER, ids=lstg4,
                              offer={
                                  'byr': True,
                                  'time': 1,
                                  'price': 3
                              })
    next = timefeats.get_feats(lstg1, time=1)
    for feat in next:
        if 'byr_offers' in feat:
            assert next[feat] == 1
        elif 'slr_offers' in feat:
            assert next[feat] == 0
        elif 'open_threads' in feat:
            assert next[feat] == 1
        elif 'open_lstgs' in feat:
            if 'cndtn' not in feat:
                assert next[feat] == 3
            else:
                assert next[feat] == 2
        elif 'byr_max' in feat:
            assert next[feat] == 3
        elif 'slr_min' in feat:
            if 'lstg' in feat:
                assert next[feat] == 5
            else:
                assert next[feat] == 5

    next = timefeats.get_feats(lstg4, time=1)
    for feat in next:
        if 'byr_offers' in feat:
            assert next[feat] == 1
        elif 'slr_offers' in feat:
            assert next[feat] == 0
        elif 'open_threads' in feat:
            assert next[feat] == 1
        elif 'open_lstgs' in feat:
            if 'cndtn' not in feat:
                assert next[feat] == 3
            else:
                assert next[feat] == 2
        elif 'byr_max' in feat:
            assert next[feat] == 2
        elif 'slr_min' in feat:
            if 'lstg' in feat:
                assert next[feat] == 5
            else:
                assert next[feat] == 5
    timefeats.update_features(trigger_type=time_triggers.SELLER_OFFER, ids=lstg1,
                              offer={
                                  'byr': False,
                                  'time': 5,
                                  'price': 4
                              })
    timefeats.update_features(trigger_type=time_triggers.SELLER_OFFER, ids=lstg4,
                              offer={
                                  'byr': False,
                                  'time': 5,
                                  'price': 4.5
                              })


def test_slr_offer(init_timefeats, lstgs):
    lstg1, lstg2, lstg3 = lstgs
    timefeats = init_timefeats
    prev = timefeats.get_feats(lstg1, time=2)
    timefeats.update_features(trigger_type=time_triggers.BUYER_OFFER, ids=lstg1,
                              offer={
                                  'byr': True,
                                  'time': 5,
                                  'price': 3
                              })
    timefeats.update_features(trigger_type=time_triggers.SELLER_OFFER, ids=lstg1,
                              offer={
                                  'byr': False,
                                  'time': 5,
                                  'price': 4
                              })
    next = timefeats.get_feats(lstg1, time=5)
    for feat in prev:
        if 'byr_offers' in feat:
            assert next[feat] == 0
        elif 'slr_offers' in feat:
            assert next[feat] == 0
        elif 'open_threads' in feat:
            assert next[feat] == 0
        elif 'open_lstgs' in feat:
            if 'cndtn' not in feat:
                assert next[feat] == 3
            else:
                assert next[feat] == 2
        elif 'byr_max' in feat:
            assert next[feat] == 0
        elif 'slr_min' in feat:
            if 'lstg' in feat:
                assert next[feat] == 5
            else:
                assert next[feat] == 5
    del lstg2['thread_id']
    next = timefeats.get_feats(lstg2, time=6)
    for feat in prev:
        if 'byr_offers' in feat:
            if 'lstg' in feat:
                assert next[feat] == 0
            else:
                assert next[feat] == 1
        elif 'slr_offers' in feat:
            if 'lstg' in feat:
                assert next[feat] == 0
            else:
                assert next[feat] == 1
        elif 'open_threads' in feat:
            if 'lstg' in feat:
                assert next[feat] == 0
            else:
                assert next[feat] == 1
        elif 'open_lstgs' in feat:
            if 'cndtn' not in feat:
                assert next[feat] == 3
            else:
                assert next[feat] == 2
        elif 'byr_max' in feat:
            if 'lstg' not in feat:
                assert next[feat] == 3
            else:
                assert next[feat] == 0
        elif 'slr_min' in feat:
            if 'lstg' in feat:
                assert next[feat] == 100
            else:
                assert next[feat] == 4
    del lstg3['thread_id']
    next = timefeats.get_feats(lstg3, 6)
    for feat in prev:
        if 'byr_offers' in feat:
            if 'lstg' in feat or 'cndtn' in feat:
                assert next[feat] == 0
            else:
                assert next[feat] == 1
        elif 'slr_offers' in feat:
            if 'lstg' in feat or 'cndtn' in feat:
                assert next[feat] == 0
            else:
                assert next[feat] == 1
        elif 'open_threads' in feat:
            if 'lstg' in feat or 'cndtn' in feat:
                print(feat)
                assert next[feat] == 0
            else:
                assert next[feat] == 1
        elif 'open_lstgs' in feat:
            if 'cndtn' not in feat:
                assert next[feat] == 3
            else:
                assert next[feat] == 1
        elif 'byr_max' in feat:
            if 'lstg' in feat or 'cndtn' in feat:
                assert next[feat] == 0
            else:
                assert next[feat] == 3
        elif 'slr_min' in feat:
            if 'lstg' in feat or 'cndtn' in feat:
                assert next[feat] == 150
            else:
                assert next[feat] == 4


import pytest
from rlenv.TimeFeatures import TimeFeatures
import rlenv.time_triggers as time_triggers
import rlenv.env_constants as consts

@pytest.fixture
def lstgs():
    thread1 = {
        'lstg': 6,
        'thread_id': 10
    }

    thread2 = {
        'lstg': 6,
        'thread_id': 11
    }

    thread3 = {
        'lstg': 6,
        'thread_id': 12
    }

    thread4 = {
        'lstg': 7,
        'thread_id': 10
    }

    thread5 = {
        'lstg': 7,
        'thread_id': 11
    }
    first_lstg = {
        'lstg': 6
    }
    second_lstg = {
        'lstg': 7
    }

    lstg1 = (thread1, thread2, thread3, first_lstg)
    lstg2 = (thread4, thread5, second_lstg)
    return lstg1, lstg2

@pytest.fixture
def timefeats():
    return TimeFeatures()

@pytest.fixture
def init_timefeats():
    timefeats = TimeFeatures()
    timefeats.initialize_time_feats(6)
    timefeats.initialize_time_feats(7)
    return timefeats


def test_initialize_feats(init_timefeats):
    timefeats = init_timefeats
    assert timefeats.lstg_active(6)
    assert timefeats.lstg_active(7)


def test_initial_feats(lstgs, init_timefeats):
    timefeats = init_timefeats
    lstg1, _ = lstgs
    thread1, thread2, thread3, lstg1 = lstg1
    feats1 = timefeats.get_feats(thread1, 0)
    feats2 = timefeats.get_feats(thread2, 0)
    feats3 = timefeats.get_feats(thread3, 0)
    feats4 = timefeats.get_feats(lstg1, 0)
    for feat in consts.TIME_FEATS:
        for feat_dict in [feats1, feats2, feats3, feats4]:
            assert feat_dict[feat] == 0


def test_first_byr_offer(lstgs, init_timefeats):
    lstg1, _ = lstgs
    thread1, thread2, thread3, lstg1 = lstg1
    timefeats = init_timefeats
    prev = timefeats.get_feats(thread1, 0)
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread1,
                              offer={
                                  'type': 'byr',
                                  'time': 5,
                                  'price': .2
                              })
    next = timefeats.get_feats(thread1, 5)
    for feat in prev:
        assert next[feat] == 0

    next = timefeats.get_feats(thread2, 6)
    for feat in prev:
        if 'byr_offers' in feat:
            assert next[feat] == 1
        elif 'slr_offers' in feat:
            assert next[feat] == 0
        elif 'byr_best' in feat:
            assert next[feat] == .2
        elif 'slr_best' in feat:
            assert next[feat] == 0
        else:
            raise RuntimeError()

    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread2,
                              offer={
                                  'byr': True,
                                  'time': 6,
                                  'price': .3
                              })
    next3 = timefeats.get_feats(thread3, 6)
    next2 = timefeats.get_feats(thread2, 6)
    next1 = timefeats.get_feats(thread1, 6)
    for feat in prev:
        if 'byr_offers' in feat:
            assert next1[feat] == 1
            assert next2[feat] == 1
            assert next3[feat] == 2
        elif 'slr_offers' in feat:
            assert next1[feat] == 0
            assert next2[feat] == 0
            assert next3[feat] == 0
        elif 'byr_best' in feat:
            assert next1[feat] == .3
            assert next2[feat] == .2
            assert next3[feat] == .3
        elif 'slr_best' in feat:
            assert next[feat] == 0
        else:
            raise RuntimeError()

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


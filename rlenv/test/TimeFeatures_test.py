import pytest
from rlenv.TimeFeatures import TimeFeatures
import rlenv.time_triggers as time_triggers
import rlenv.env_consts as consts

@pytest.fixture
def lstgs():
    thread1 = (6, 10)
    thread2 = (6, 11)
    thread3 = (6, 12)
    thread4 = (7, 10)
    thread5 = (7, 11)
    first_lstg = (6, )
    second_lstg = (7, )

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
                                  'type': 'byr',
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
            assert next1[feat] == pytest.approx(.3)
            assert next2[feat] == pytest.approx(.2)
            assert next3[feat] == pytest.approx(.3)
        elif 'slr_best' in feat:
            assert next1[feat] == 0
            assert next2[feat] == 0
            assert next3[feat] == 0
        else:
            raise RuntimeError()


def test_first_slr_offer(init_timefeats, lstgs):
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
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread2,
                              offer={
                                  'type': 'byr',
                                  'time': 6,
                                  'price': .3
                              })
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread1,
                              offer={
                                  'type': 'slr',
                                  'time': 7,
                                  'price': 1-.8,
                              })
    next3 = timefeats.get_feats(thread3, 6)
    next2 = timefeats.get_feats(thread2, 6)
    next1 = timefeats.get_feats(thread1, 6)
    for feat in prev:
        if 'byr_offers' in feat:
            if 'open' in feat:
                assert next1[feat] == 1
                assert next2[feat] == 0
                assert next3[feat] == 1
            else:
                assert next1[feat] == 1
                assert next2[feat] == 1
                assert next3[feat] == 2
        elif 'slr_offers' in feat:
            assert next1[feat] == 0
            assert next2[feat] == 1
            assert next3[feat] == 1
        elif 'byr_best' in feat:
            if 'open' in feat:
                assert next1[feat] == pytest.approx(.3)
                assert next2[feat] == pytest.approx(0)
                assert next3[feat] == pytest.approx(.3)
            else:
                assert next1[feat] == pytest.approx(.3)
                assert next2[feat] == pytest.approx(.2)
                assert next3[feat] == pytest.approx(.3)
        elif 'slr_best' in feat:
            assert next1[feat] == pytest.approx(0)
            assert next2[feat] == pytest.approx(.2)
            assert next3[feat] == pytest.approx(.2)
        else:
            raise RuntimeError()
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread2,
                              offer={
                                  'type': 'slr',
                                  'time': 7,
                                  'price': 1 - .7,
                              })
    next3 = timefeats.get_feats(thread3, 8)
    next2 = timefeats.get_feats(thread2, 8)
    next1 = timefeats.get_feats(thread1, 8)
    for feat in prev:
        if 'byr_offers' in feat:
            if 'open' in feat:
                assert next1[feat] == 0
                assert next2[feat] == 0
                assert next3[feat] == 0
            else:
                assert next1[feat] == 1
                assert next2[feat] == 1
                assert next3[feat] == 2
        elif 'slr_offers' in feat:
            assert next1[feat] == 1
            assert next2[feat] == 1
            assert next3[feat] == 2
        elif 'byr_best' in feat:
            if 'open' in feat:
                assert next1[feat] == pytest.approx(0)
                assert next2[feat] == pytest.approx(0)
                assert next3[feat] == pytest.approx(0)
            else:
                assert next1[feat] == pytest.approx(.3)
                assert next2[feat] == pytest.approx(.2)
                assert next3[feat] == pytest.approx(.3)
        elif 'slr_best' in feat:
            assert next1[feat] == pytest.approx(.3)
            assert next2[feat] == pytest.approx(.2)
            assert next3[feat] == pytest.approx(.3)
        else:
            raise RuntimeError()


def test_byr_offer(init_timefeats, lstgs):
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
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread2,
                              offer={
                                  'type': 'byr',
                                  'time': 6,
                                  'price': .3
                              })
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread1,
                              offer={
                                  'type': 'slr',
                                  'time': 7,
                                  'price': 1-.8,
                              })
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread2,
                              offer={
                                  'type': 'slr',
                                  'time': 7,
                                  'price': 1 - .7,
                              })
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread1,
                              offer={
                                  'type': 'byr',
                                  'time': 5,
                                  'price': .4
                              })
    next3 = timefeats.get_feats(thread3, 6)
    next2 = timefeats.get_feats(thread2, 6)
    next1 = timefeats.get_feats(thread1, 6)
    for feat in prev:
        if 'byr_offers' in feat:
            if 'open' in feat:
                assert next1[feat] == 0
                assert next2[feat] == 1
                assert next3[feat] == 1
            else:
                assert next1[feat] == 1
                assert next2[feat] == 2
                assert next3[feat] == 3
        elif 'slr_offers' in feat:
            if 'open' in feat:
                assert next1[feat] == 1
                assert next2[feat] == 0
                assert next3[feat] == 1
            else:
                assert next1[feat] == 1
                assert next2[feat] == 1
                assert next3[feat] == 2
        elif 'byr_best' in feat:
            if 'open' in feat:
                assert next1[feat] == pytest.approx(0)
                assert next2[feat] == pytest.approx(.4)
                assert next3[feat] == pytest.approx(.4)
            else:
                assert next1[feat] == pytest.approx(.3)
                assert next2[feat] == pytest.approx(.4)
                assert next3[feat] == pytest.approx(.4)
        elif 'slr_best' in feat:
            if 'open' in feat:
                assert next1[feat] == pytest.approx(.3)
                assert next2[feat] == pytest.approx(0)
                assert next3[feat] == pytest.approx(.3)
            else:
                assert next1[feat] == pytest.approx(.3)
                assert next2[feat] == pytest.approx(.2)
                assert next3[feat] == pytest.approx(.3)
        else:
            raise RuntimeError()
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread2,
                              offer={
                                  'type': 'byr',
                                  'time': 8,
                                  'price': .35
                              })
    next3 = timefeats.get_feats(thread3, 6)
    next2 = timefeats.get_feats(thread2, 6)
    next1 = timefeats.get_feats(thread1, 6)
    for feat in prev:
        if 'byr_offers' in feat:
            if 'open' in feat:
                assert next1[feat] == 1
                assert next2[feat] == 1
                assert next3[feat] == 2
            else:
                assert next1[feat] == 2
                assert next2[feat] == 2
                assert next3[feat] == 4
        elif 'slr_offers' in feat:
            if 'open' in feat:
                assert next1[feat] == 0
                assert next2[feat] == 0
                assert next3[feat] == 0
            else:
                assert next1[feat] == 1
                assert next2[feat] == 1
                assert next3[feat] == 2
        elif 'byr_best' in feat:
            if 'open' in feat:
                assert next1[feat] == pytest.approx(.35)
                assert next2[feat] == pytest.approx(.4)
                assert next3[feat] == pytest.approx(.4)
            else:
                assert next1[feat] == pytest.approx(.35)
                assert next2[feat] == pytest.approx(.4)
                assert next3[feat] == pytest.approx(.4)
        elif 'slr_best' in feat:
            if 'open' in feat:
                assert next1[feat] == pytest.approx(0)
                assert next2[feat] == pytest.approx(0)
                assert next3[feat] == pytest.approx(0)
            else:
                assert next1[feat] == pytest.approx(.3)
                assert next2[feat] == pytest.approx(.2)
                assert next3[feat] == pytest.approx(.3)
        else:
            raise RuntimeError()


def test_slr_offer(init_timefeats, lstgs):
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
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread2,
                              offer={
                                  'type': 'byr',
                                  'time': 6,
                                  'price': .3
                              })
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread1,
                              offer={
                                  'type': 'slr',
                                  'time': 7,
                                  'price': 1-.8,
                              })
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread2,
                              offer={
                                  'type': 'slr',
                                  'time': 7,
                                  'price': 1 - .7,
                              })
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread1,
                              offer={
                                  'type': 'byr',
                                  'time': 5,
                                  'price': .4
                              })
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread2,
                              offer={
                                  'type': 'byr',
                                  'time': 8,
                                  'price': .35
                              })
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread1,
                              offer={
                                  'type': 'slr',
                                  'time': 9,
                                  'price': 1-.6
                              })
    next3 = timefeats.get_feats(thread3, 6)
    next2 = timefeats.get_feats(thread2, 6)
    next1 = timefeats.get_feats(thread1, 6)
    for feat in prev:
        if 'byr_offers' in feat:
            if 'open' in feat:
                assert next1[feat] == 1
                assert next2[feat] == 0
                assert next3[feat] == 1
            else:
                assert next1[feat] == 2
                assert next2[feat] == 2
                assert next3[feat] == 4
        elif 'slr_offers' in feat:
            if 'open' in feat:
                assert next1[feat] == 0
                assert next2[feat] == 1
                assert next3[feat] == 1
            else:
                assert next1[feat] == 1
                assert next2[feat] == 2
                assert next3[feat] == 3
        elif 'byr_best' in feat:
            if 'open' in feat:
                assert next1[feat] == pytest.approx(.35)
                assert next2[feat] == pytest.approx(0)
                assert next3[feat] == pytest.approx(.35)
            else:
                assert next1[feat] == pytest.approx(.35)
                assert next2[feat] == pytest.approx(.4)
                assert next3[feat] == pytest.approx(.4)
        elif 'slr_best' in feat:
            if 'open' in feat:
                assert next1[feat] == pytest.approx(0)
                assert next2[feat] == pytest.approx(.4)
                assert next3[feat] == pytest.approx(.4)
            else:
                assert next1[feat] == pytest.approx(.3)
                assert next2[feat] == pytest.approx(.4)
                assert next3[feat] == pytest.approx(.4)
        else:
            raise RuntimeError()
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread2,
                              offer={
                                  'type': 'slr',
                                  'time': 9,
                                  'price': 1 - .65
                              })
    next3 = timefeats.get_feats(thread3, 6)
    next2 = timefeats.get_feats(thread2, 6)
    next1 = timefeats.get_feats(thread1, 6)
    for feat in prev:
        if 'byr_offers' in feat:
            if 'open' in feat:
                assert next1[feat] == 0
                assert next2[feat] == 0
                assert next3[feat] == 0
            else:
                assert next1[feat] == 2
                assert next2[feat] == 2
                assert next3[feat] == 4
        elif 'slr_offers' in feat:
            if 'open' in feat:
                assert next1[feat] == 1
                assert next2[feat] == 1
                assert next3[feat] == 2
            else:
                assert next1[feat] == 2
                assert next2[feat] == 2
                assert next3[feat] == 4
        elif 'byr_best' in feat:
            if 'open' in feat:
                assert next1[feat] == pytest.approx(0)
                assert next2[feat] == pytest.approx(0)
                assert next3[feat] == pytest.approx(0)
            else:
                assert next1[feat] == pytest.approx(.35)
                assert next2[feat] == pytest.approx(.4)
                assert next3[feat] == pytest.approx(.4)
        elif 'slr_best' in feat:
            if 'open' in feat:
                assert next1[feat] == pytest.approx(.35)
                assert next2[feat] == pytest.approx(.4)
                assert next3[feat] == pytest.approx(.4)
            else:
                assert next1[feat] == pytest.approx(.35)
                assert next2[feat] == pytest.approx(.4)
                assert next3[feat] == pytest.approx(.4)
        else:
            raise RuntimeError()


def test_byr_rejection(init_timefeats, lstgs):
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
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread2,
                              offer={
                                  'type': 'byr',
                                  'time': 6,
                                  'price': .3
                              })
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread1,
                              offer={
                                  'type': 'slr',
                                  'time': 7,
                                  'price': 1-.8,
                              })
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread2,
                              offer={
                                  'type': 'slr',
                                  'time': 7,
                                  'price': 1 - .7,
                              })
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread1,
                              offer={
                                  'type': 'byr',
                                  'time': 5,
                                  'price': .4
                              })
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread2,
                              offer={
                                  'type': 'byr',
                                  'time': 8,
                                  'price': .35
                              })
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread1,
                              offer={
                                  'type': 'slr',
                                  'time': 9,
                                  'price': 1-.6
                              })
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread2,
                              offer={
                                  'type': 'slr',
                                  'time': 9,
                                  'price': 1 - .65
                              })
    timefeats.update_features(trigger_type=time_triggers.BYR_REJECTION, ids=thread1)

    next3 = timefeats.get_feats(thread3, 6)
    next2 = timefeats.get_feats(thread2, 6)
    next1 = timefeats.get_feats(thread1, 6)
    for feat in prev:
        if 'byr_offers' in feat:
            if 'open' in feat:
                assert next1[feat] == 0
                assert next2[feat] == 0
                assert next3[feat] == 0
            else:
                assert next1[feat] == 2
                assert next2[feat] == 2
                assert next3[feat] == 4
        elif 'slr_offers' in feat:
            if 'open' in feat:
                assert next1[feat] == 1
                assert next2[feat] == 0
                assert next3[feat] == 1
            else:
                assert next1[feat] == 2
                assert next2[feat] == 2
                assert next3[feat] == 4
        elif 'byr_best' in feat:
            if 'open' in feat:
                assert next1[feat] == pytest.approx(0)
                assert next2[feat] == pytest.approx(0)
                assert next3[feat] == pytest.approx(0)
            else:
                assert next1[feat] == pytest.approx(.35)
                assert next2[feat] == pytest.approx(.4)
                assert next3[feat] == pytest.approx(.4)
        elif 'slr_best' in feat:
            if 'open' in feat:
                assert next1[feat] == pytest.approx(.35)
                assert next2[feat] == pytest.approx(0)
                assert next3[feat] == pytest.approx(.35)
            else:
                assert next1[feat] == pytest.approx(.35)
                assert next2[feat] == pytest.approx(.4)
                assert next3[feat] == pytest.approx(.4)
        else:
            raise RuntimeError()
    timefeats.update_features(trigger_type=time_triggers.BYR_REJECTION, ids=thread2)
    next3 = timefeats.get_feats(thread3, 6)
    next2 = timefeats.get_feats(thread2, 6)
    next1 = timefeats.get_feats(thread1, 6)
    for feat in prev:
        if 'byr_offers' in feat:
            if 'open' in feat:
                assert next1[feat] == 0
                assert next2[feat] == 0
                assert next3[feat] == 0
            else:
                assert next1[feat] == 2
                assert next2[feat] == 2
                assert next3[feat] == 4
        elif 'slr_offers' in feat:
            if 'open' in feat:
                assert next1[feat] == 0
                assert next2[feat] == 0
                assert next3[feat] == 0
            else:
                assert next1[feat] == 2
                assert next2[feat] == 2
                assert next3[feat] == 4
        elif 'byr_best' in feat:
            if 'open' in feat:
                assert next1[feat] == pytest.approx(0)
                assert next2[feat] == pytest.approx(0)
                assert next3[feat] == pytest.approx(0)
            else:
                assert next1[feat] == pytest.approx(.35)
                assert next2[feat] == pytest.approx(.4)
                assert next3[feat] == pytest.approx(.4)
        elif 'slr_best' in feat:
            if 'open' in feat:
                assert next1[feat] == pytest.approx(0)
                assert next2[feat] == pytest.approx(0)
                assert next3[feat] == pytest.approx(0)
            else:
                assert next1[feat] == pytest.approx(.35)
                assert next2[feat] == pytest.approx(.4)
                assert next3[feat] == pytest.approx(.4)
        else:
            raise RuntimeError()


def test_slr_rejection_early(init_timefeats, lstgs):
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
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread2,
                              offer={
                                  'type': 'byr',
                                  'time': 6,
                                  'price': .3
                              })
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread1,
                              offer={
                                  'type': 'slr',
                                  'time': 7,
                                  'price': 1-.8,
                              })
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread2,
                              offer={
                                  'type': 'slr',
                                  'time': 7,
                                  'price': 1 - .7,
                              })
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread1,
                              offer={
                                  'type': 'byr',
                                  'time': 5,
                                  'price': .4
                              })
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread2,
                              offer={
                                  'type': 'byr',
                                  'time': 8,
                                  'price': .35
                              })
    timefeats.update_features(trigger_type=time_triggers.SLR_REJECTION, ids=thread1,
                              offer={
                                  'type': 'slr',
                                  'time': 7,
                                  'price': 1 - .8,
                              })

    next3 = timefeats.get_feats(thread3, 6)
    next2 = timefeats.get_feats(thread2, 6)
    next1 = timefeats.get_feats(thread1, 6)
    for feat in prev:
        if 'byr_offers' in feat:
            if 'open' in feat:
                assert next1[feat] == 1
                assert next2[feat] == 0
                assert next3[feat] == 1
            else:
                assert next1[feat] == 2
                assert next2[feat] == 2
                assert next3[feat] == 4
        elif 'slr_offers' in feat:
            if 'open' in feat:
                assert next1[feat] == 0
                assert next2[feat] == 1
                assert next3[feat] == 1
            else:
                assert next1[feat] == 1
                assert next2[feat] == 1
                assert next3[feat] == 2
        elif 'byr_best' in feat:
            if 'open' in feat:
                assert next1[feat] == pytest.approx(.35)
                assert next2[feat] == pytest.approx(0)
                assert next3[feat] == pytest.approx(.35)
            else:
                assert next1[feat] == pytest.approx(.35)
                assert next2[feat] == pytest.approx(.4)
                assert next3[feat] == pytest.approx(.4)
        elif 'slr_best' in feat:
            if 'open' in feat:
                assert next1[feat] == pytest.approx(0)
                assert next2[feat] == pytest.approx(.2)
                assert next3[feat] == pytest.approx(.2)
            else:
                assert next1[feat] == pytest.approx(.3)
                assert next2[feat] == pytest.approx(.2)
                assert next3[feat] == pytest.approx(.3)
        else:
            raise RuntimeError()

    print(' slr offers: %d' % next1['slr_offers'])
    timefeats.update_features(trigger_type=time_triggers.SLR_REJECTION, ids=thread2,
                              offer={
                                  'type': 'slr',
                                  'time': 7,
                                  'price': 1 - .7,
                              })

    next3 = timefeats.get_feats(thread3, 6)
    next2 = timefeats.get_feats(thread2, 6)
    next1 = timefeats.get_feats(thread1, 6)
    for feat in prev:
        if 'byr_offers' in feat:
            if 'open' in feat:
                assert next1[feat] == 0
                assert next2[feat] == 0
                assert next3[feat] == 0
            else:
                assert next1[feat] == 2
                assert next2[feat] == 2
                assert next3[feat] == 4
        elif 'slr_offers' in feat:
            if 'open' in feat:
                assert next1[feat] == 1
                assert next2[feat] == 1
                assert next3[feat] == 2
            else:
                assert next1[feat] == 1
                assert next2[feat] == 1
                assert next3[feat] == 2
        elif 'byr_best' in feat:
            if 'open' in feat:
                assert next1[feat] == pytest.approx(0)
                assert next2[feat] == pytest.approx(0)
                assert next3[feat] == pytest.approx(0)
            else:
                assert next1[feat] == pytest.approx(.35)
                assert next2[feat] == pytest.approx(.4)
                assert next3[feat] == pytest.approx(.4)
        elif 'slr_best' in feat:
            if 'open' in feat:
                assert next1[feat] == pytest.approx(.3)
                assert next2[feat] == pytest.approx(.2)
                assert next3[feat] == pytest.approx(.3)
            else:
                assert next1[feat] == pytest.approx(.3)
                assert next2[feat] == pytest.approx(.2)
                assert next3[feat] == pytest.approx(.3)
        else:
            raise RuntimeError()


def test_buyer_acceptance(lstgs, init_timefeats):
    lstg1, lstg2 = lstgs
    thread1, thread2, thread3, lstg1 = lstg1
    thread4, thread5, lstg2 = lstg2
    timefeats = init_timefeats
    prev = timefeats.get_feats(thread1, 0)
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread1,
                              offer={
                                  'type': 'byr',
                                  'time': 5,
                                  'price': .2
                              })
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread2,
                              offer={
                                  'type': 'byr',
                                  'time': 6,
                                  'price': .3
                              })
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread1,
                              offer={
                                  'type': 'slr',
                                  'time': 7,
                                  'price': 1 - .8,
                              })
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread2,
                              offer={
                                  'type': 'slr',
                                  'time': 7,
                                  'price': 1 - .7,
                              })
    timefeats.update_features(trigger_type=time_triggers.SALE, ids=thread1)
    assert not timefeats.lstg_active(thread1[0])
    assert not timefeats.lstg_active(thread2[0])
    assert not timefeats.lstg_active(thread3[0])
    assert not timefeats.lstg_active(lstg1[0])
    print(lstg2)
    assert timefeats.lstg_active(lstg2[0])
    with pytest.raises(RuntimeError):
        timefeats.get_feats(lstg1, 9)
    next4 = timefeats.get_feats(thread4, 9)
    next5 = timefeats.get_feats(thread5, 9)
    next_lstg = timefeats.get_feats(thread5, 9)
    for feat in next_lstg:
        assert next4[feat] == 0
        assert next5[feat] == 0
        assert next_lstg[feat] == 0


def test_slr_acceptance(lstgs, init_timefeats):
    lstg1, lstg2 = lstgs
    thread1, thread2, thread3, lstg1 = lstg1
    thread4, thread5, lstg2 = lstg2
    timefeats = init_timefeats
    prev = timefeats.get_feats(thread1, 0)
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread1,
                              offer={
                                  'type': 'byr',
                                  'time': 5,
                                  'price': .2
                              })
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread2,
                              offer={
                                  'type': 'byr',
                                  'time': 6,
                                  'price': .3
                              })
    timefeats.update_features(trigger_type=time_triggers.SALE, ids=thread1)
    assert not timefeats.lstg_active(thread1[0])
    assert not timefeats.lstg_active(thread2[0])
    assert not timefeats.lstg_active(thread3[0])
    assert not timefeats.lstg_active(lstg1[0])
    assert timefeats.lstg_active(lstg2[0])
    with pytest.raises(RuntimeError):
        timefeats.get_feats(lstg1, 9)
    next4 = timefeats.get_feats(thread4, 9)
    next5 = timefeats.get_feats(thread5, 9)
    next_lstg = timefeats.get_feats(thread5, 9)
    for feat in next_lstg:
        assert next4[feat] == 0
        assert next5[feat] == 0
        assert next_lstg[feat] == 0


def test_lstg_expiration(lstgs, init_timefeats):
    lstg1, lstg2 = lstgs
    thread1, thread2, thread3, lstg1 = lstg1
    thread4, thread5, lstg2 = lstg2
    timefeats = init_timefeats
    prev = timefeats.get_feats(thread1, 0)
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread1,
                              offer={
                                  'type': 'byr',
                                  'time': 5,
                                  'price': .2
                              })
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread2,
                              offer={
                                  'type': 'byr',
                                  'time': 6,
                                  'price': .3
                              })
    timefeats.update_features(trigger_type=time_triggers.OFFER, ids=thread1,
                              offer={
                                  'type': 'slr',
                                  'time': 7,
                                  'price': 1 - .8,
                              })
    timefeats.update_features(trigger_type=time_triggers.LSTG_EXPIRATION,
                              ids=lstg1)

    assert not timefeats.lstg_active(thread1[0])
    assert not timefeats.lstg_active(thread2[0])
    assert not timefeats.lstg_active(thread3[0])
    assert not timefeats.lstg_active(lstg1[0])
    assert timefeats.lstg_active(lstg2[0])
    with pytest.raises(RuntimeError):
        timefeats.get_feats(lstg1, 9)
    with pytest.raises(RuntimeError):
        timefeats.get_feats(thread1, 9)
    timefeats.update_features(trigger_type=time_triggers.LSTG_EXPIRATION,
                              ids=lstg2)
    with pytest.raises(RuntimeError):
        timefeats.get_feats(lstg2, 9)
    assert not timefeats.lstg_active(lstg2[0])
    assert not timefeats.lstg_active(thread5[0])
    assert not timefeats.lstg_active(thread4[0])
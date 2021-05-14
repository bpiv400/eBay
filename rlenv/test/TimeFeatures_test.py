import pytest
import torch
import numpy as np
import rlenv.const
from featnames import TIME_FEATS, THREAD_COUNT
from constants import MAX_DELAY_TURN
from rlenv.time.TimeFeatures import TimeFeatures


@pytest.fixture
def lstgs():
    thread1 = 1
    thread2 = 2
    lstg1 = thread1, thread2
    return lstg1


@pytest.fixture
def timefeats():
    return TimeFeatures()


@pytest.fixture
def init_timefeats():
    timefeats = TimeFeatures()
    return timefeats


def compare(actual, exp):
    """
    Approximate equality between expected tensor and actual tensor

    :param actual: 1 dimensional np.array
    :param exp: 1 dimensional torch.tensor
    :return: NA
    """
    if len(actual.shape) == 0:
        actual = np.array([actual])
    actual = torch.from_numpy(actual).float()
    assert torch.all(torch.lt(torch.abs(torch.add(actual, -exp)), 1e-6))


def test_initialize_feats(init_timefeats):
    timefeats = init_timefeats


def test_initial_feats(lstgs, init_timefeats):
    timefeats = init_timefeats
    thread1, thread2, thread3 = lstgs
    feats1 = timefeats.get_feats(thread1, 0)
    feats2 = timefeats.get_feats(thread2, 0)
    feats3 = timefeats.get_feats(thread3, 0)
    exp = torch.zeros(len(TIME_FEATS))
    compare(feats1, exp)
    compare(feats2, exp)
    compare(feats3, exp)


def test_first_byr_offer(lstgs, init_timefeats):
    timefeats = init_timefeats
    thread1, thread2, thread3 = lstgs
    prev = timefeats.get_feats(thread1, 0)
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread1,
                                  offer={
                                  'type': 'byr',
                                  'time': 5,
                                  'price': .2
                              })
    next = timefeats.get_feats(thread1, 5)
    exp = torch.zeros(len(TIME_FEATS)).float()
    compare(next, exp)

    next = timefeats.get_feats(thread2, 6)
    exp = torch.tensor([0, 0, 0, 0, 0, 0, 1, .2, 1, .2, 1, .2, 1]).float()
    compare(next, exp)

    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread2,
                                  offer={
                                  'type': 'byr',
                                  'time': 6,
                                  'price': .3
                              })
    next3 = timefeats.get_feats(thread3, 6)
    exp3 = torch.tensor([0, 0, 0, 0, 0, 0, 2, .3, 2, .3, 2, .3, 2]).float()
    compare(next3, exp3)

    next2 = timefeats.get_feats(thread2, 6)
    exp2 = torch.tensor([0, 0, 0, 0, 0, 0, 1, .2, 1, .2, 1, .2, 1]).float()
    compare(next, exp2)

    next1 = timefeats.get_feats(thread1, 6)
    exp1 = torch.tensor([0, 0, 0, 0, 0, 0, 1, .3, 1, .3, 1, .3, 1]).float()
    compare(next1, exp1)


def test_first_slr_offer(init_timefeats, lstgs):
    timefeats = init_timefeats
    thread1, thread2, thread3 = lstgs
    timefeats = init_timefeats
    prev = timefeats.get_feats(thread1, 0)
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread1,
                                  offer={
                                  'type': 'byr',
                                  'time': 5,
                                  'price': .2
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread2,
                                  offer={
                                  'type': 'byr',
                                  'time': 6,
                                  'price': .3
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread1,
                                  offer={
                                  'type': 'slr',
                                  'time': 7,
                                  'price': 1-.8,
                              })
    next3 = timefeats.get_feats(thread3, 6)
    next2 = timefeats.get_feats(thread2, 6)
    next1 = timefeats.get_feats(thread1, 6)
    for i, feat in enumerate(TIME_FEATS):
        if 'byr_offers' in feat:
            if 'open' in feat:
                assert next1[i] == 1
                assert next2[i] == 0
                assert next3[i] == 1
            else:
                assert next1[i] == 1
                assert next2[i] == 1
                assert next3[i] == 2
        elif 'slr_offers' in feat:
            assert next1[i] == 0
            assert next2[i] == 1
            assert next3[i] == 1
        elif 'byr_best' in feat:
            if 'open' in feat:
                compare(next1[i], torch.tensor(.3).float())
                compare(next2[i], torch.tensor(0).float())
                compare(next3[i], torch.tensor(.3).float())
            else:
                compare(next1[i], torch.tensor(.3).float())
                compare(next2[i], torch.tensor(.2).float())
                compare(next3[i], torch.tensor(.3).float())
        elif 'slr_best' in feat:
            compare(next1[i], torch.tensor(0))
            compare(next2[i], torch.tensor(.2).float())
            compare(next3[i], torch.tensor(.2).float())
        elif feat == THREAD_COUNT:
            compare(next3[i], torch.tensor(2).float())
            compare(next1[i], torch.tensor(1).float())
            compare(next2[i], torch.tensor(1).float())
        else:
            raise RuntimeError()

    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread2,
                                  offer={
                                  'type': 'slr',
                                  'time': 7,
                                  'price': 1 - .7,
                              })
    next3 = timefeats.get_feats(thread3, 8)
    next2 = timefeats.get_feats(thread2, 8)
    next1 = timefeats.get_feats(thread1, 8)
    for i, feat in enumerate(TIME_FEATS):
        if 'byr_offers' in feat:
            if 'open' in feat:
                assert next1[i] == 0
                assert next2[i] == 0
                assert next3[i] == 0
            else:
                assert next1[i] == 1
                assert next2[i] == 1
                assert next3[i] == 2
        elif 'slr_offers' in feat:
            assert next1[i] == 1
            assert next2[i] == 1
            assert next3[i] == 2
        elif 'byr_best' in feat:
            if 'open' in feat:
                compare(next1[i], torch.tensor(0).float())
                compare(next2[i], torch.tensor(0).float())
                compare(next3[i], torch.tensor(0).float())
            else:
                compare(next1[i], torch.tensor(.3).float())
                compare(next2[i], torch.tensor(.2).float())
                compare(next3[i], torch.tensor(.3).float())
        elif 'slr_best' in feat:
            compare(next1[i], torch.tensor(.3).float())
            compare(next2[i], torch.tensor(.2).float())
            compare(next3[i], torch.tensor(.3).float())
        elif feat == THREAD_COUNT:
            compare(next3[i], torch.tensor(2).float())
            compare(next1[i], torch.tensor(1).float())
            compare(next2[i], torch.tensor(1).float())
        else:
            raise RuntimeError()


def test_byr_offer(init_timefeats, lstgs):
    timefeats = init_timefeats
    thread1, thread2, thread3 = lstgs
    timefeats = init_timefeats
    prev = timefeats.get_feats(thread1, 0)
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread1,
                                  offer={
                                  'type': 'byr',
                                  'time': 5,
                                  'price': .2
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread2,
                                  offer={
                                  'type': 'byr',
                                  'time': 6,
                                  'price': .3
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread1,
                                  offer={
                                  'type': 'slr',
                                  'time': 7,
                                  'price': 1-.8,
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread2,
                                  offer={
                                  'type': 'slr',
                                  'time': 7,
                                  'price': 1 - .7,
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread1,
                                  offer={
                                  'type': 'byr',
                                  'time': 5,
                                  'price': .4
                              })
    next3 = timefeats.get_feats(thread3, 6)
    next2 = timefeats.get_feats(thread2, 6)
    next1 = timefeats.get_feats(thread1, 6)
    for i, feat in enumerate(TIME_FEATS):
        if 'byr_offers' in feat:
            if 'open' in feat:
                assert next1[i] == 0
                assert next2[i] == 1
                assert next3[i] == 1
            else:
                assert next1[i] == 1
                assert next2[i] == 2
                assert next3[i] == 3
        elif 'slr_offers' in feat:
            if 'open' in feat:
                assert next1[i] == 1
                assert next2[i] == 0
                assert next3[i] == 1
            else:
                assert next1[i] == 1
                assert next2[i] == 1
                assert next3[i] == 2
        elif 'byr_best' in feat:
            if 'open' in feat:
                compare(next1[i], torch.tensor(0).float())
                compare(next2[i], torch.tensor(.4).float())
                compare(next3[i], torch.tensor(.4).float())
            else:
                compare(next1[i], torch.tensor(.3).float())
                compare(next2[i], torch.tensor(.4).float())
                compare(next3[i], torch.tensor(.4).float())
        elif 'slr_best' in feat:
            if 'open' in feat:
                compare(next1[i], torch.tensor(.3).float())
                compare(next2[i], torch.tensor(0).float())
                compare(next3[i], torch.tensor(.3).float())
            else:
                compare(next1[i], torch.tensor(.3).float())
                compare(next2[i], torch.tensor(.2).float())
                compare(next3[i], torch.tensor(.3).float())
        elif feat == THREAD_COUNT:
            compare(next3[i], torch.tensor(2).float())
            compare(next1[i], torch.tensor(1).float())
            compare(next2[i], torch.tensor(1).float())
        else:
            raise RuntimeError()

    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread2,
                                  offer={
                                  'type': 'byr',
                                  'time': 8,
                                  'price': .35
                              })
    next3 = timefeats.get_feats(thread3, 6)
    next2 = timefeats.get_feats(thread2, 6)
    next1 = timefeats.get_feats(thread1, 6)
    for i, feat in enumerate(TIME_FEATS):
        if 'byr_offers' in feat:
            if 'open' in feat:
                assert next1[i] == 1
                assert next2[i] == 1
                assert next3[i] == 2
            else:
                assert next1[i] == 2
                assert next2[i] == 2
                assert next3[i] == 4
        elif 'slr_offers' in feat:
            if 'open' in feat:
                assert next1[i] == 0
                assert next2[i] == 0
                assert next3[i] == 0
            else:
                assert next1[i] == 1
                assert next2[i] == 1
                assert next3[i] == 2
        elif 'byr_best' in feat:
            if 'open' in feat:
                compare(next1[i], torch.tensor(.35).float())
                compare(next2[i], torch.tensor(.4).float())
                compare(next3[i], torch.tensor(.4).float())
            else:
                compare(next1[i], torch.tensor(.35).float())
                compare(next2[i], torch.tensor(.4).float())
                compare(next3[i], torch.tensor(.4).float())
        elif 'slr_best' in feat:
            if 'open' in feat:
                compare(next1[i], torch.tensor(0).float())
                compare(next2[i], torch.tensor(0).float())
                compare(next3[i], torch.tensor(0).float())
            else:
                compare(next1[i], torch.tensor(.3).float())
                compare(next2[i], torch.tensor(.2).float())
                compare(next3[i], torch.tensor(.3).float())
        elif feat == THREAD_COUNT:
            compare(next3[i], torch.tensor(2).float())
            compare(next1[i], torch.tensor(1).float())
            compare(next2[i], torch.tensor(1).float())
        else:
            raise RuntimeError()


def test_slr_offer(init_timefeats, lstgs):
    timefeats = init_timefeats
    thread1, thread2, thread3 = lstgs
    timefeats = init_timefeats
    prev = timefeats.get_feats(thread1, 0)
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread1,
                                  offer={
                                  'type': 'byr',
                                  'time': 5,
                                  'price': .2
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread2,
                                  offer={
                                  'type': 'byr',
                                  'time': 6,
                                  'price': .3
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread1,
                                  offer={
                                  'type': 'slr',
                                  'time': 7,
                                  'price': 1-.8,
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread2,
                                  offer={
                                  'type': 'slr',
                                  'time': 7,
                                  'price': 1 - .7,
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread1,
                                  offer={
                                  'type': 'byr',
                                  'time': 5,
                                  'price': .4
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread2,
                                  offer={
                                  'type': 'byr',
                                  'time': 8,
                                  'price': .35
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread1,
                                  offer={
                                  'type': 'slr',
                                  'time': 9,
                                  'price': 1-.6
                              })
    next3 = timefeats.get_feats(thread3, 6)
    next2 = timefeats.get_feats(thread2, 6)
    next1 = timefeats.get_feats(thread1, 6)
    for i, feat in enumerate(TIME_FEATS):
        if 'byr_offers' in feat:
            if 'open' in feat:
                assert next1[i] == 1
                assert next2[i] == 0
                assert next3[i] == 1
            else:
                assert next1[i] == 2
                assert next2[i] == 2
                assert next3[i] == 4
        elif 'slr_offers' in feat:
            if 'open' in feat:
                assert next1[i] == 0
                assert next2[i] == 1
                assert next3[i] == 1
            else:
                assert next1[i] == 1
                assert next2[i] == 2
                assert next3[i] == 3
        elif 'byr_best' in feat:
            if 'open' in feat:
                compare(next1[i], torch.tensor(.35).float())
                compare(next2[i], torch.tensor(0).float())
                compare(next3[i], torch.tensor(.35).float())
            else:
                compare(next1[i], torch.tensor(.35).float())
                compare(next2[i], torch.tensor(.4).float())
                compare(next3[i], torch.tensor(.4).float())
        elif 'slr_best' in feat:
            if 'open' in feat:
                compare(next1[i], torch.tensor(0).float())
                compare(next2[i], torch.tensor(.4).float())
                compare(next3[i], torch.tensor(.4).float())
            else:
                compare(next1[i], torch.tensor(.3).float())
                compare(next2[i], torch.tensor(.4).float())
                compare(next3[i], torch.tensor(.4).float())
        elif feat == THREAD_COUNT:
            compare(next3[i], torch.tensor(2).float())
            compare(next1[i], torch.tensor(1).float())
            compare(next2[i], torch.tensor(1).float())
        else:
            raise RuntimeError()
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread2,
                                  offer={
                                  'type': 'slr',
                                  'time': 9,
                                  'price': 1 - .65
                              })
    next3 = timefeats.get_feats(thread3, 6)
    next2 = timefeats.get_feats(thread2, 6)
    next1 = timefeats.get_feats(thread1, 6)
    for i, feat in enumerate(TIME_FEATS):
        if 'byr_offers' in feat:
            if 'open' in feat:
                assert next1[i] == 0
                assert next2[i] == 0
                assert next3[i] == 0
            else:
                assert next1[i] == 2
                assert next2[i] == 2
                assert next3[i] == 4
        elif 'slr_offers' in feat:
            if 'open' in feat:
                assert next1[i] == 1
                assert next2[i] == 1
                assert next3[i] == 2
            else:
                assert next1[i] == 2
                assert next2[i] == 2
                assert next3[i] == 4
        elif 'byr_best' in feat:
            if 'open' in feat:
                compare(next1[i], torch.tensor(0).float())
                compare(next2[i], torch.tensor(0).float())
                compare(next3[i], torch.tensor(0).float())
            else:
                compare(next1[i], torch.tensor(.35).float())
                compare(next2[i], torch.tensor(.4).float())
                compare(next3[i], torch.tensor(.4).float())
        elif 'slr_best' in feat:
            if 'open' in feat:
                compare(next1[i], torch.tensor(.35).float())
                compare(next2[i], torch.tensor(.4).float())
                compare(next3[i], torch.tensor(.4).float())
            else:
                compare(next1[i], torch.tensor(.35).float())
                compare(next2[i], torch.tensor(.4).float())
                compare(next3[i], torch.tensor(.4).float())
        elif feat == THREAD_COUNT:
            compare(next3[i], torch.tensor(2).float())
            compare(next1[i], torch.tensor(1).float())
            compare(next2[i], torch.tensor(1).float())
        else:
            raise RuntimeError()


def test_byr_rejection(init_timefeats, lstgs):
    timefeats = init_timefeats
    thread1, thread2, thread3 = lstgs
    timefeats = init_timefeats
    prev = timefeats.get_feats(thread1, 0)
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread1,
                                  offer={
                                  'type': 'byr',
                                  'time': 5,
                                  'price': .2
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread2,
                                  offer={
                                  'type': 'byr',
                                  'time': 6,
                                  'price': .3
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread1,
                                  offer={
                                  'type': 'slr',
                                  'time': 7,
                                  'price': 1-.8,
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread2,
                                  offer={
                                  'type': 'slr',
                                  'time': 7,
                                  'price': 1 - .7,
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread1,
                                  offer={
                                  'type': 'byr',
                                  'time': 5,
                                  'price': .4
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread2,
                                  offer={
                                  'type': 'byr',
                                  'time': 8,
                                  'price': .35
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread1,
                                  offer={
                                  'type': 'slr',
                                  'time': 9,
                                  'price': 1-.6
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread2,
                                  offer={
                                  'type': 'slr',
                                  'time': 9,
                                  'price': 1 - .65
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.BYR_REJECTION, thread_id=thread1,
                                  offer={'time': 9})

    next3 = timefeats.get_feats(thread3, 9)
    next2 = timefeats.get_feats(thread2, 9)
    next1 = timefeats.get_feats(thread1, 9)
    for i, feat in enumerate(TIME_FEATS):
        if 'byr_offers' in feat:
            if 'open' in feat:
                assert next1[i] == 0
                assert next2[i] == 0
                assert next3[i] == 0
            else:
                assert next1[i] == 2
                assert next2[i] == 2
                assert next3[i] == 4
        elif 'slr_offers' in feat:
            if 'open' in feat:
                assert next1[i] == 1
                assert next2[i] == 0
                assert next3[i] == 1
            else:
                assert next1[i] == 2
                assert next2[i] == 2
                assert next3[i] == 4
        elif 'byr_best' in feat:
            if 'open' in feat:
                compare(next1[i], torch.tensor(0).float())
                compare(next2[i], torch.tensor(0).float())
                compare(next3[i], torch.tensor(0).float())
            else:
                compare(next1[i], torch.tensor(.35).float())
                compare(next2[i], torch.tensor(.4).float())
                compare(next3[i], torch.tensor(.4).float())
        elif 'slr_best' in feat:
            if 'open' in feat:
                compare(next1[i], torch.tensor(.35).float())
                compare(next2[i], torch.tensor(0).float())
                compare(next3[i], torch.tensor(.35).float())
            else:
                compare(next1[i], torch.tensor(.35).float())
                compare(next2[i], torch.tensor(.4).float())
                compare(next3[i], torch.tensor(.4).float())
        elif feat == THREAD_COUNT:
            compare(next3[i], torch.tensor(2).float())
            compare(next1[i], torch.tensor(1).float())
            compare(next2[i], torch.tensor(1).float())
        else:
            raise RuntimeError()

    timefeats.update_features_old(trigger_type=rlenv.const.BYR_REJECTION, thread_id=thread2,
                                  offer={'time': 9})
    next3 = timefeats.get_feats(thread3, 9)
    next2 = timefeats.get_feats(thread2, 9)
    next1 = timefeats.get_feats(thread1, 9)
    for i, feat in enumerate(TIME_FEATS):
        if 'byr_offers' in feat:
            if 'open' in feat:
                assert next1[i] == 0
                assert next2[i] == 0
                assert next3[i] == 0
            else:
                assert next1[i] == 2
                assert next2[i] == 2
                assert next3[i] == 4
        elif 'slr_offers' in feat:
            if 'open' in feat:
                assert next1[i] == 0
                assert next2[i] == 0
                assert next3[i] == 0
            else:
                assert next1[i] == 2
                assert next2[i] == 2
                assert next3[i] == 4
        elif 'byr_best' in feat:
            if 'open' in feat:
                compare(next1[i], torch.tensor(0).float())
                compare(next2[i], torch.tensor(0).float())
                compare(next3[i], torch.tensor(0).float())
            else:
                compare(next1[i], torch.tensor(.35).float())
                compare(next2[i], torch.tensor(.4).float())
                compare(next3[i], torch.tensor(.4).float())
        elif 'slr_best' in feat:
            if 'open' in feat:
                compare(next1[i], torch.tensor(0).float())
                compare(next2[i], torch.tensor(0).float())
                compare(next3[i], torch.tensor(0).float())
            else:
                compare(next1[i], torch.tensor(.35).float())
                compare(next2[i], torch.tensor(.4).float())
                compare(next3[i], torch.tensor(.4).float())
        elif feat == THREAD_COUNT:
            compare(next3[i], torch.tensor(2).float())
            compare(next1[i], torch.tensor(1).float())
            compare(next2[i], torch.tensor(1).float())
        else:
            raise RuntimeError()


def test_slr_rejection_early(init_timefeats, lstgs):
    timefeats = init_timefeats
    thread1, thread2, thread3 = lstgs
    timefeats = init_timefeats
    prev = timefeats.get_feats(thread1, 0)
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread1,
                                  offer={
                                  'type': 'byr',
                                  'time': 5,
                                  'price': .2
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread2,
                                  offer={
                                  'type': 'byr',
                                  'time': 6,
                                  'price': .3
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread1,
                                  offer={
                                  'type': 'slr',
                                  'time': 7,
                                  'price': 1-.8,
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread2,
                                  offer={
                                  'type': 'slr',
                                  'time': 7,
                                  'price': 1 - .7,
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread1,
                                  offer={
                                  'type': 'byr',
                                  'time': 5,
                                  'price': .4
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread2,
                                  offer={
                                  'type': 'byr',
                                  'time': 8,
                                  'price': .35
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.SLR_REJECTION, thread_id=thread1,
                                  offer={
                                  'type': 'slr',
                                  'time': 7,
                                  'price': 1 - .8,
                              })

    next3 = timefeats.get_feats(thread3, 6)
    next2 = timefeats.get_feats(thread2, 6)
    next1 = timefeats.get_feats(thread1, 6)
    for i, feat in enumerate(TIME_FEATS):
        if 'byr_offers' in feat:
            if 'open' in feat:
                assert next1[i] == 1
                assert next2[i] == 0
                assert next3[i] == 1
            else:
                assert next1[i] == 2
                assert next2[i] == 2
                assert next3[i] == 4
        elif 'slr_offers' in feat:
            if 'open' in feat:
                assert next1[i] == 0
                assert next2[i] == 1
                assert next3[i] == 1
            else:
                assert next1[i] == 1
                assert next2[i] == 1
                assert next3[i] == 2
        elif 'byr_best' in feat:
            if 'open' in feat:
                compare(next1[i], torch.tensor(.35).float())
                compare(next2[i], torch.tensor(0).float())
                compare(next3[i], torch.tensor(.35).float())
            else:
                compare(next1[i], torch.tensor(.35).float())
                compare(next2[i], torch.tensor(.4).float())
                compare(next3[i], torch.tensor(.4).float())
        elif 'slr_best' in feat:
            if 'open' in feat:
                compare(next1[i], torch.tensor(0).float())
                compare(next2[i], torch.tensor(.2).float())
                compare(next3[i], torch.tensor(.2).float())
        elif feat == THREAD_COUNT:
            compare(next3[i], torch.tensor(2).float())
            compare(next1[i], torch.tensor(1).float())
            compare(next2[i], torch.tensor(1).float())
        else:
            raise RuntimeError()
    timefeats.update_features_old(trigger_type=rlenv.const.SLR_REJECTION, thread_id=thread2,
                                  offer={
                                  'type': 'slr',
                                  'time': 7,
                                  'price': 1 - .7,
                              })

    next3 = timefeats.get_feats(thread3, 6)
    next2 = timefeats.get_feats(thread2, 6)
    next1 = timefeats.get_feats(thread1, 6)
    for i, feat in enumerate(TIME_FEATS):
        if 'byr_offers' in feat:
            if 'open' in feat:
                assert next1[i] == 0
                assert next2[i] == 0
                assert next3[i] == 0
            else:
                assert next1[i] == 2
                assert next2[i] == 2
                assert next3[i] == 4
        elif 'slr_offers' in feat:
            if 'open' in feat:
                assert next1[i] == 1
                assert next2[i] == 1
                assert next3[i] == 2
            else:
                assert next1[i] == 1
                assert next2[i] == 1
                assert next3[i] == 2
        elif 'byr_best' in feat:
            if 'open' in feat:
                compare(next1[i], torch.tensor(0).float())
                compare(next2[i], torch.tensor(0).float())
                compare(next3[i], torch.tensor(0).float())
            else:
                compare(next1[i], torch.tensor(.35).float())
                compare(next2[i], torch.tensor(.4).float())
                compare(next3[i], torch.tensor(.4).float())
        elif 'slr_best' in feat:
            if 'open' in feat:
                compare(next1[i], torch.tensor(.3).float())
                compare(next2[i], torch.tensor(.2).float())
                compare(next3[i], torch.tensor(.3).float())
            else:
                compare(next1[i], torch.tensor(.3).float())
                compare(next2[i], torch.tensor(.2).float())
                compare(next3[i], torch.tensor(.3).float())
        elif feat == THREAD_COUNT:
            compare(next3[i], torch.tensor(2).float())
            compare(next1[i], torch.tensor(1).float())
            compare(next2[i], torch.tensor(1).float())
        else:
            raise RuntimeError()


def test_byr_offer_best_expire_no_backup(init_timefeats, lstgs):
    thread1, thread2, thread3 = lstgs
    timefeats = init_timefeats
    prev = timefeats.get_feats(thread1, 0)
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread1,
                                  offer={
                                  'type': 'byr',
                                  'time': 5,
                                  'price': .2
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread2,
                                  offer={
                                  'type': 'byr',
                                  'time': 6,
                                  'price': .3
                              })
    next2 = timefeats.get_feats(thread2, 6)
    next1 = timefeats.get_feats(thread1, 6)
    next3 = timefeats.get_feats(thread3, 6)
    for i, feat in enumerate(TIME_FEATS):
        if 'byr_offers' in feat:
            if 'open' in feat:
                assert next1[i] == 1
                assert next2[i] == 1
                assert next3[i] == 2
            else:
                assert next1[i] == 1
                assert next2[i] == 1
                assert next3[i] == 2
        elif 'slr_offers' in feat:
            assert next1[i] == 0
            assert next2[i] == 0
            assert next3[i] == 0
        elif 'byr_best' in feat:
            if 'open' in feat:
                compare(next1[i], torch.tensor(.3).float())
                compare(next2[i], torch.tensor(.2).float())
                compare(next3[i], torch.tensor(.3).float())
            else:
                compare(next1[i], torch.tensor(.3).float())
                compare(next2[i], torch.tensor(.2).float())
                compare(next3[i], torch.tensor(.3).float())
        elif 'slr_best' in feat:
            compare(next1[i], torch.tensor(0))
            compare(next2[i], torch.tensor(0).float())
            compare(next3[i], torch.tensor(0).float())
        elif feat == THREAD_COUNT:
            compare(next1[i], torch.tensor(1).float())
            compare(next2[i], torch.tensor(1).float())
            compare(next3[i], torch.tensor(2).float())
        else:
            raise RuntimeError()

    next1 = timefeats.get_feats(thread1, 8 + MAX_DELAY_TURN)
    next2 = timefeats.get_feats(thread2, 8 + MAX_DELAY_TURN)
    next3 = timefeats.get_feats(thread3, 8 + MAX_DELAY_TURN)

    for i, feat in enumerate(TIME_FEATS):
        if 'byr_offers' in feat:
            if 'open' in feat:
                assert next1[i] == 1
                assert next2[i] == 1
                assert next3[i] == 2
            elif 'recent' in feat:
                assert next1[i] == 0
                assert next2[i] == 0
                assert next3[i] == 0
            else:
                assert next1[i] == 1
                assert next2[i] == 1
                assert next3[i] == 2
        elif 'slr_offers' in feat:
            assert next1[i] == 0
            assert next2[i] == 0
            assert next3[i] == 0
        elif 'byr_best' in feat:
            if 'open' in feat:
                compare(next1[i], torch.tensor(.3).float())
                compare(next2[i], torch.tensor(.2).float())
                compare(next3[i], torch.tensor(.3).float())
            elif 'recent' in feat:
                compare(next1[i], torch.tensor(0).float())
                compare(next2[i], torch.tensor(0).float())
                compare(next3[i], torch.tensor(0).float())
        elif 'slr_best' in feat:
            compare(next1[i], torch.tensor(0).float())
            compare(next2[i], torch.tensor(0).float())
            compare(next3[i], torch.tensor(0).float())
        elif feat == THREAD_COUNT:
            compare(next3[i], torch.tensor(2).float())
            compare(next1[i], torch.tensor(1).float())
            compare(next2[i], torch.tensor(1).float())
        else:
            raise RuntimeError()


def test_byr_best_expire_with_backup(init_timefeats, lstgs):
    thread1, thread2, thread3 = lstgs
    timefeats = init_timefeats
    prev = timefeats.get_feats(thread1, 0)
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread1,
                                  offer={
                                  'type': 'byr',
                                  'time': 5,
                                  'price': .2
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread2,
                                  offer={
                                  'type': 'byr',
                                  'time': 6,
                                  'price': .3
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread1,
                                  offer={
                                  'type': 'slr',
                                  'time': 15,
                                  'price': .25
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread1,
                                  offer={
                                  'type': 'byr',
                                  'time': 20,
                                  'price': .18
                              })


    next1 = timefeats.get_feats(thread1, 16 + MAX_DELAY_TURN)
    next2 = timefeats.get_feats(thread2, 16 + MAX_DELAY_TURN)
    next3 = timefeats.get_feats(thread3, 16 + MAX_DELAY_TURN)

    for i, feat in enumerate(TIME_FEATS):
        if 'byr_offers' in feat:
            if 'open' in feat:
                assert next1[i] == 1
                assert next2[i] == 1
                assert next3[i] == 2
            elif 'recent' in feat:
                assert next1[i] == 0
                assert next2[i] == 1
                assert next3[i] == 1
            else:
                assert next1[i] == 1
                assert next2[i] == 2
                assert next3[i] == 3
        elif 'slr_offers' in feat:
            if 'open' in feat:
                assert next1[i] == 0
                assert next2[i] == 0
                assert next3[i] == 0
            elif 'recent' in feat:
                assert next1[i] == 0
                assert next2[i] == 0
                assert next3[i] == 0
            else:
                assert next1[i] == 0
                assert next2[i] == 1
                assert next3[i] == 1
        elif 'byr_best' in feat:
            if 'open' in feat:
                compare(next1[i], torch.tensor(.3).float())
                compare(next2[i], torch.tensor(.18).float())
                compare(next3[i], torch.tensor(.3).float())
            elif 'recent' in feat:
                compare(next1[i], torch.tensor(0).float())
                compare(next2[i], torch.tensor(.18).float())
                compare(next3[i], torch.tensor(.18).float())
        elif 'slr_best' in feat:
            if 'open' in feat:
                compare(next1[i], torch.tensor(0).float())
                compare(next2[i], torch.tensor(0).float())
                compare(next3[i], torch.tensor(0).float())
            elif 'recent' in feat:
                compare(next1[i], torch.tensor(0).float())
                compare(next2[i], torch.tensor(0).float())
                compare(next3[i], torch.tensor(0).float())
            else:
                compare(next1[i], torch.tensor(0).float())
                compare(next2[i], torch.tensor(.25).float())
                compare(next3[i], torch.tensor(.25).float())
        elif feat == THREAD_COUNT:
            compare(next3[i], torch.tensor(2).float())
            compare(next1[i], torch.tensor(1).float())
            compare(next2[i], torch.tensor(1).float())
        else:
            raise RuntimeError()


def test_overriding_recent_max(init_timefeats, lstgs):
    thread1, thread2, thread3 = lstgs
    timefeats = init_timefeats
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread1,
                                  offer={
                                  'type': 'byr',
                                  'time': 5,
                                  'price': .2
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread2,
                                  offer={
                                  'type': 'byr',
                                  'time': 25,
                                  'price': .35
                              })

    next1 = timefeats.get_feats(thread1, 16 + MAX_DELAY_TURN)
    next2 = timefeats.get_feats(thread2, 16 + MAX_DELAY_TURN)
    next3 = timefeats.get_feats(thread3, 16 + MAX_DELAY_TURN)

    for i, feat in enumerate(TIME_FEATS):
        if 'byr_offers' in feat:
            if 'recent' in feat:
                assert next1[i] == 1
                assert next2[i] == 0
                assert next3[i] == 1
        elif 'slr_offers' in feat:
            if 'recent' in feat:
                assert next1[i] == 0
                assert next2[i] == 0
                assert next3[i] == 0
        elif 'byr_best' in feat:
            if 'recent' in feat:
                compare(next1[i], torch.tensor(.35).float())
                compare(next2[i], torch.tensor(0).float())
                compare(next3[i], torch.tensor(.35).float())
        elif 'slr_best' in feat:
            if 'recent' in feat:
                compare(next1[i], torch.tensor(0).float())
                compare(next2[i], torch.tensor(0).float())
                compare(next3[i], torch.tensor(0).float())


def test_overriding_recent_max_same_thread(init_timefeats, lstgs):
    thread1, thread2, thread3 = lstgs
    timefeats = init_timefeats
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread1,
                                  offer={
                                  'type': 'byr',
                                  'time': 5,
                                  'price': .2
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread2,
                                  offer={
                                  'type': 'byr',
                                  'time': 25,
                                  'price': .18
                              })
    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread1,
                                  offer={
                                    'type': 'slr',
                                    'time': 35,
                                    'price': .3
                              })

    timefeats.update_features_old(trigger_type=rlenv.const.OFFER, thread_id=thread1,
                                  offer={
                                  'type': 'byr',
                                  'time': 40,
                                  'price': .35
                              })

    next1 = timefeats.get_feats(thread1, 16 + MAX_DELAY_TURN)
    next2 = timefeats.get_feats(thread2, 16 + MAX_DELAY_TURN)
    next3 = timefeats.get_feats(thread3, 16 + MAX_DELAY_TURN)

    for i, feat in enumerate(TIME_FEATS):
        if 'byr_offers' in feat:
            if 'recent' in feat:
                assert next1[i] == 1
                assert next2[i] == 1
                assert next3[i] == 2
        elif 'slr_offers' in feat:
            if 'recent' in feat:
                assert next1[i] == 0
                assert next2[i] == 1
                assert next3[i] == 1
        elif 'byr_best' in feat:
            if 'recent' in feat:
                compare(next1[i], torch.tensor(.18).float())
                compare(next2[i], torch.tensor(.35).float())
                compare(next3[i], torch.tensor(.35).float())
        elif 'slr_best' in feat:
            if 'recent' in feat:
                compare(next1[i], torch.tensor(0).float())
                compare(next2[i], torch.tensor(.3).float())
                compare(next3[i], torch.tensor(.3).float())
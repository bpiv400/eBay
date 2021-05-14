import pytest
from processing.b_feats.tf import get_lstg_time_feats
from processing.b_feats.test.test_utils import (compare_all,
                                                get_exp_feats,
                                                update)
from rlenv.const import SLR_REJECTION, BYR_REJECTION, OFFER
from rlenv.time.TimeFeatures import TimeFeatures

# consts
COLS = ['accept', 'clock', 'reject', 'byr',
        'start_price', 'norm', 'price', 'censored', 'message']
ACCEPTANCE = 'acceptance' # time trigger for df not needed for environment


@pytest.fixture()
def timefeats():
    return TimeFeatures()


def test_interwoven_byr_rejection(timefeats):
    offer = {
        'type': 'byr',
        'price': .5,
        'time': 4,
    }
    events = add_event(None, trigger_type=OFFER, thread_id=1, offer=offer)
    timefeats.update_features(trigger_type=OFFER, thread_id=1, offer=offer)
    t1_o1 = timefeats.get_feats(thread_id=None, time=4)
    t2_o1 = timefeats.get_feats(thread_id=None, time=4)
    offer['price'] = .65
    offer['time'] = 6
    events = add_event(events, trigger_type=OFFER, thread_id=2, offer=offer)
    timefeats.update_features(trigger_type=OFFER, thread_id=2, offer=offer)
    t1_o2 = timefeats.get_feats(thread_id=None, time=6)
    t2_o2 = timefeats.get_feats(thread_id=None, time=6)
    offer['price'] = .9
    offer['time'] = 8
    offer['type'] = 'slr'
    events = add_event(events, trigger_type=OFFER, thread_id=2, offer=offer)
    timefeats.update_features(trigger_type=OFFER, thread_id=2, offer=offer)
    t1_o3 = timefeats.get_feats(thread_id=None, time=8)
    t2_o3 = timefeats.get_feats(thread_id=None, time=8)
    offer['price'] = .85
    offer['time'] = 9
    events = add_event(events, trigger_type=OFFER, thread_id=1, offer=offer)
    timefeats.update_features(trigger_type=OFFER, thread_id=1, offer=offer)
    t1_o4 = timefeats.get_feats(thread_id=None, time=9)
    t2_o4 = timefeats.get_feats(thread_id=None, time=9)
    # rejection
    offer['price'] = .65
    offer['time'] = 10
    offer['type'] = 'byr'
    events = add_event(events, trigger_type=BYR_REJECTION, thread_id=2, offer=offer)
    timefeats.update_features(trigger_type=BYR_REJECTION, thread_id=2)
    t1_o5 = timefeats.get_feats(thread_id=None, time=10)
    offer['time'] = 11
    offer['price'] = .5
    events = add_event(events, trigger_type=BYR_REJECTION, thread_id=1, offer=offer)
    print('')
    print(events)
    act = get_lstg_time_feats(events, full=True)
    print(act.loc[:, ['byr_offers_open', 'byr_best_open', 'byr_best', 'byr_offers']])
    print(act.loc[:, ['slr_offers_open', 'slr_best_open', 'slr_best', 'slr_offers']])
    compare(act.loc[(0, 4)].values, t1_o1)
    compare(act.loc[(0, 4)].values, t2_o1)
    compare(act.loc[(0, 6)].values, t1_o2)
    compare(act.loc[(0, 6)].values, t2_o2)
    compare(act.loc[(0, 8)].values, t2_o3)
    compare(act.loc[(0, 8)].values, t1_o3)
    compare(act.loc[(0, 9)].values, t2_o4)
    compare(act.loc[(0, 9)].values, t1_o4)
    compare(act.loc[(0, 10)].values, t1_o5)


def test_two_lstgs_interwoven_two_threads(timefeats):
    offer = {
        'type': 'byr',
        'price': .5,
        'time': 4,
    }
    time_checks = list()
    exp = list()
    events = update(None, timefeats, thread_id=1, offer=offer, trigger_type=OFFER,
                    lstg=0)
    events = update(events, None, thread_id=1, offer=offer, trigger_type=OFFER,
                    lstg=1)
    get_exp_feats((1, 4), timefeats, exp, time_checks)

    offer['type'] = 'slr'
    offer['price'] = .1
    offer['time'] = 5
    events = update(events, timefeats, thread_id=1, offer=offer, trigger_type=OFFER, lstg=0)
    events = update(events, None, thread_id=1, offer=offer, trigger_type=OFFER, lstg=1)
    get_exp_feats((1, 5), timefeats, exp, time_checks)

    offer['type'] = 'byr'
    offer['price'] = .6
    offer['time'] = 6
    events = update(events, timefeats, thread_id=1, offer=offer, trigger_type=OFFER, lstg=0)
    events = update(events, None, thread_id=1, offer=offer, trigger_type=OFFER, lstg=1)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'slr'
    offer['price'] = .2
    offer['time'] = 7
    events = update(events, timefeats, thread_id=1, offer=offer, trigger_type=OFFER, lstg=0)
    events = update(events, None, thread_id=1, offer=offer, trigger_type=OFFER, lstg=1)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'byr'
    offer['price'] = .4
    offer['time'] = 8
    events = update(events, timefeats, thread_id=2, offer=offer, trigger_type=OFFER, lstg=0)
    events = update(events, None, thread_id=2, offer=offer, trigger_type=OFFER, lstg=1)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'slr'
    offer['price'] = .2
    offer['time'] = 9
    events = update(events, timefeats, thread_id=2, offer=offer, trigger_type=OFFER, lstg=0)
    events = update(events, None, thread_id=2, offer=offer, trigger_type=OFFER, lstg=1)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'byr'
    offer['price'] = .65
    offer['time'] = 10
    events = update(events, timefeats, thread_id=3, offer=offer, trigger_type=OFFER, lstg=0)
    events = update(events, None, thread_id=3, offer=offer, trigger_type=OFFER, lstg=1)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((3, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'slr'
    offer['price'] = .2
    offer['time'] = 11
    events = update(events, timefeats, thread_id=3, offer=offer, trigger_type=OFFER, lstg=0)
    events = update(events, None, thread_id=3, offer=offer, trigger_type=OFFER, lstg=1)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((3, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'byr'
    offer['price'] = .6
    offer['time'] = 12
    events = update(events, timefeats, thread_id=1, offer=offer, lstg=0,
                    trigger_type=BYR_REJECTION)
    events = update(events, None, thread_id=1, offer=offer, lstg=1,
                    trigger_type=BYR_REJECTION)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((3, offer['time']), timefeats, exp, time_checks)

    offer['price'] = .4
    offer['time'] = 13
    events = update(events, timefeats, thread_id=2, offer=offer, lstg=0,
                    trigger_type=BYR_REJECTION)
    events = update(events, None, thread_id=2, offer=offer, lstg=1,
                    trigger_type=BYR_REJECTION)
    get_exp_feats((3, offer['time']), timefeats, exp, time_checks)

    offer['price'] = .8
    offer['time'] = 14
    events = update(events, timefeats, thread_id=3, offer=offer, lstg=0,
                    trigger_type=ACCEPTANCE)
    events = update(events, None, thread_id=3, offer=offer, lstg=1,
                    trigger_type=ACCEPTANCE)
    compare_all(events, exp, time_checks, lstg=0)
    compare_all(events, exp, time_checks, lstg=1)


def test_slr_auto_reject_offer_while_slr_offer_outstanding(timefeats):
    offer = {
        'type': 'byr',
        'price': .5,
        'time': 4,
    }
    time_checks = list()
    exp = list()
    events = update(None, timefeats, offer=offer, trigger_type=OFFER, thread_id=1)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)

    offer['time'] = 5
    offer['price'] = .55
    events = update(events, timefeats, offer=offer, trigger_type=OFFER, thread_id=2)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)


    offer['time'] = 6
    offer['price'] = .3
    offer['type'] = 'slr'
    events = update(events, timefeats, offer=offer, trigger_type=OFFER, thread_id=1)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)

    offer['time'] = 7
    offer['price'] = .4
    offer['type'] = 'slr'
    events = update(events, timefeats, offer=offer, trigger_type=OFFER, thread_id=2)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)

    offer['time'] = 8
    offer['price'] = .6
    offer['type'] = 'byr'
    events = update(events, timefeats, offer=offer, trigger_type=OFFER, thread_id=1)
    offer['type'] = 'slr'
    offer['price'] = .3
    events = update(events, timefeats, offer=offer, trigger_type=SLR_REJECTION, thread_id=1)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((3, offer['time']), timefeats, exp, time_checks)

    offer['price'] = .1
    offer['type'] = 'byr'
    offer['time'] = 9
    events = update(events, timefeats, offer=offer, trigger_type=OFFER, thread_id=3)
    offer['type'] = 'slr'
    offer['price'] = 0
    events = update(events, timefeats, offer=offer, trigger_type=SLR_REJECTION, thread_id=3)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((3, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'byr'
    offer['price'] = .6
    offer['time'] = 10
    events = update(events, timefeats, offer=offer, trigger_type=ACCEPTANCE, thread_id=1)
    compare_all(events, exp, time_checks)

def test_slr_auto_reject_offer_while_byr_offer_outstanding(timefeats):
    offer = {
        'type': 'byr',
        'price': .5,
        'time': 4,
    }
    time_checks = list()
    exp = list()
    events = update(None, timefeats, offer=offer, trigger_type=OFFER, thread_id=1)

    offer['time'] = 5
    offer['price'] = .4
    events = update(events, timefeats, offer=offer, trigger_type=OFFER, thread_id=2)
    offer['type'] = 'slr'
    offer['price'] = 0
    events = update(events, timefeats, offer=offer, trigger_type=SLR_REJECTION, thread_id=2)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)

    offer['time'] = 6
    offer['type'] = 'slr'
    offer['price'] = .5
    events = update(events, timefeats, offer=offer, trigger_type=ACCEPTANCE, thread_id=1)
    compare_all(events, exp, time_checks)


def test_same_time_byr_offer(timefeats):
    offer = {
        'type': 'byr',
        'price': .5,
        'time': 4,
    }
    time_checks = list()
    exp = list()
    events = update(None, timefeats, offer=offer, trigger_type=OFFER, thread_id=1)
    offer['price'] = .6
    events = update(events, timefeats, offer=offer, trigger_type=OFFER, thread_id=2)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)

    offer['time'] = 5
    offer['type'] = 'slr'
    offer['price'] = 0
    events = update(events, timefeats, offer=offer, trigger_type=SLR_REJECTION, thread_id=1)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)

    offer['price'] = .4
    offer['time'] = 6
    events = update(events, timefeats, offer=offer, trigger_type=ACCEPTANCE, thread_id=2)
    compare_all(events, exp, time_checks)


def test_sequential_rej_byr_accept(timefeats):
    offer = {
        'type': 'byr',
        'price': .5,
        'time': 4,
    }
    time_checks = list()
    exp = list()
    events = add_event(None, trigger_type=OFFER, thread_id=1, offer=offer)
    timefeats.update_features(trigger_type=OFFER, thread_id=1, offer=offer)
    get_exp_feats((1, 4), timefeats, exp, time_checks)
    offer['price'] = .2
    offer['time'] = 6
    offer['type'] = 'slr'
    events = add_event(events, trigger_type=OFFER, thread_id=1, offer=offer)
    timefeats.update_features(trigger_type=OFFER, thread_id=1, offer=offer)
    get_exp_feats((1, 6), timefeats, exp, time_checks)

    offer['type'] = 'byr'
    offer['price'] = .63
    offer['time'] = 7
    events = add_event(events, trigger_type=OFFER, thread_id=1, offer=offer)
    timefeats.update_features(trigger_type=OFFER, thread_id=1, offer=offer)
    get_exp_feats((1, 7), timefeats, exp, time_checks)
    offer['type'] = 'slr'
    offer['price'] = .3
    offer['time'] = 8
    events = add_event(events, trigger_type=OFFER, thread_id=1, offer=offer)
    timefeats.update_features(trigger_type=OFFER, thread_id=1, offer=offer)
    get_exp_feats((1, 8), timefeats, exp, time_checks)

    offer['type'] = 'byr'
    offer['price'] = .63
    offer['time'] = 9
    events = add_event(events, trigger_type=BYR_REJECTION, thread_id=1, offer=offer)
    timefeats.update_features(trigger_type=BYR_REJECTION, thread_id=1, offer=offer)
    get_exp_feats((1, 9), timefeats, exp, time_checks)

    offer['time'] = 10
    offer['price'] = .6
    events = add_event(events, trigger_type=OFFER, thread_id=2, offer=offer)
    timefeats.update_features(trigger_type=OFFER, thread_id=2, offer=offer)
    get_exp_feats((2, 10), timefeats, exp, time_checks)

    offer['time'] = 11
    offer['type'] = 'slr'
    offer['price'] = .3
    events = update(events, timefeats, thread_id=2, offer=offer, trigger_type=OFFER)
    get_exp_feats((2, 11), timefeats, exp, time_checks)

    # acceptance
    offer['price'] = .7
    offer['time'] = 12
    offer['type'] = 'byr'
    events = add_event(events, trigger_type=ACCEPTANCE, thread_id=2, offer=offer)
    print('events input')
    print(events)
    compare_all(events, exp, time_checks)


def test_partial_seq_partial_overlap(timefeats):
    offer = {
        'type': 'byr',
        'price': .5,
        'time': 4,
    }
    time_checks = list()
    exp = list()
    events = update(None, timefeats, thread_id=1, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, 4), timefeats, exp, time_checks)

    offer['type'] = 'slr'
    offer['price'] = .1
    offer['time'] = 5
    events = update(events, timefeats, thread_id=1, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, 5), timefeats, exp, time_checks)

    offer['type'] = 'byr'
    offer['price'] = .6
    offer['time'] = 6
    events = update(events, timefeats, thread_id=1, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'slr'
    offer['price'] = .2
    offer['time'] = 7
    events = update(events, timefeats, thread_id=1, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'byr'
    offer['price'] = .4
    offer['time'] = 8
    events = update(events, timefeats, thread_id=2, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'slr'
    offer['price'] = .2
    offer['time'] = 9
    events = update(events, timefeats, thread_id=2, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'byr'
    offer['price'] = .65
    offer['time'] = 10
    events = update(events, timefeats, thread_id=3, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((3, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'slr'
    offer['price'] = .2
    offer['time'] = 11
    events = update(events, timefeats, thread_id=3, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((3, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'byr'
    offer['price'] = .6
    offer['time'] = 12
    events = update(events, timefeats, thread_id=1, offer=offer,
                    trigger_type=BYR_REJECTION)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((3, offer['time']), timefeats, exp, time_checks)

    offer['price'] = .4
    offer['time'] = 13
    events = update(events, timefeats, thread_id=2, offer=offer,
                    trigger_type=BYR_REJECTION)
    get_exp_feats((3, offer['time']), timefeats, exp, time_checks)

    offer['price'] = .8
    offer['time'] = 14
    events = update(events, timefeats, thread_id=3, offer=offer,
                    trigger_type=ACCEPTANCE)
    compare_all(events, exp, time_checks)


def test_slr_reject(timefeats):
    offer = {
        'type': 'byr',
        'price': .5,
        'time': 4,
    }
    time_checks = list()
    exp = list()
    events = update(None, timefeats, thread_id=1, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, 4), timefeats, exp, time_checks)

    offer['type'] = 'slr'
    offer['price'] = .1
    offer['time'] = 5
    events = update(events, timefeats, thread_id=1, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, 5), timefeats, exp, time_checks)

    offer['type'] = 'byr'
    offer['price'] = .6
    offer['time'] = 6
    events = update(events, timefeats, thread_id=1, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'slr'
    offer['price'] = .1
    offer['time'] = 7
    events = update(events, timefeats, thread_id=1, offer=offer, trigger_type=SLR_REJECTION)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'byr'
    offer['price'] = .4
    offer['time'] = 8
    events = update(events, timefeats, thread_id=2, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'slr'
    offer['price'] = 0
    offer['time'] = 9
    events = update(events, timefeats, thread_id=2, offer=offer, trigger_type=SLR_REJECTION)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'byr'
    offer['price'] = .65
    offer['time'] = 10
    events = update(events, timefeats, thread_id=3, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((3, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'slr'
    offer['price'] = 0
    offer['time'] = 11
    events = update(events, timefeats, thread_id=3, offer=offer, trigger_type=SLR_REJECTION)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((3, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'byr'
    offer['price'] = .6
    offer['time'] = 12
    events = update(events, timefeats, thread_id=1, offer=offer,
                    trigger_type=BYR_REJECTION)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((3, offer['time']), timefeats, exp, time_checks)

    offer['price'] = .4
    offer['time'] = 13
    events = update(events, timefeats, thread_id=2, offer=offer,
                    trigger_type=BYR_REJECTION)
    get_exp_feats((3, offer['time']), timefeats, exp, time_checks)

    offer['price'] = 1
    offer['time'] = 14
    events = update(events, timefeats, thread_id=3, offer=offer,
                    trigger_type=ACCEPTANCE)
    compare_all(events, exp, time_checks)


def test_slr_accept_worst_buyer(timefeats):
    offer = {
        'type': 'byr',
        'price': .5,
        'time': 4,
    }
    time_checks = list()
    exp = list()
    events = update(None, timefeats, thread_id=1, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, 4), timefeats, exp, time_checks)
    get_exp_feats((2, 4), timefeats, exp, time_checks)

    offer['type'] = 'byr'
    offer['price'] = .60
    offer['time'] = 5
    events = update(events, timefeats, thread_id=2, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, 5), timefeats, exp, time_checks)
    get_exp_feats((2, 5), timefeats, exp, time_checks)
    get_exp_feats((3, 5), timefeats, exp, time_checks)

    offer['type'] = 'byr'
    offer['price'] = .65
    offer['time'] = 6
    events = update(events, timefeats, thread_id=3, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((3, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'slr'
    offer['price'] = 0
    offer['time'] = 7
    events = update(events, timefeats, thread_id=3, offer=offer, trigger_type=SLR_REJECTION)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((3, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'slr'
    offer['price'] = 0
    offer['time'] = 8
    events = update(events, timefeats, thread_id=2, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((3, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'slr'
    offer['price'] = .5
    offer['time'] = 9
    events = update(events, timefeats, thread_id=1, offer=offer, trigger_type=ACCEPTANCE)
    compare_all(events, exp, time_checks)



# buyer acceptance
# late buyer rejection
# seller acceptance
# fully sequential threads



def test_slr_reject_expire(timefeats):
    offer = {
        'type': 'byr',
        'price': .5,
        'time': 4,
    }
    time_checks = list()
    exp = list()
    events = update(None, timefeats, thread_id=1, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, 4), timefeats, exp, time_checks)

    offer['type'] = 'slr'
    offer['price'] = .1
    offer['time'] = 5
    events = update(events, timefeats, thread_id=1, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, 5), timefeats, exp, time_checks)

    offer['type'] = 'byr'
    offer['price'] = .6
    offer['time'] = 6 + EXPIRATION
    events = update(events, timefeats, thread_id=1, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'slr'
    offer['price'] = .1
    offer['time'] = 7 + EXPIRATION
    events = update(events, timefeats, thread_id=1, offer=offer, trigger_type=SLR_REJECTION)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'byr'
    offer['price'] = .4
    offer['time'] = 8 + EXPIRATION
    events = update(events, timefeats, thread_id=2, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'slr'
    offer['price'] = 0
    offer['time'] = 9 + EXPIRATION
    events = update(events, timefeats, thread_id=2, offer=offer, trigger_type=SLR_REJECTION)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'byr'
    offer['price'] = .65
    offer['time'] = 10 + EXPIRATION
    events = update(events, timefeats, thread_id=3, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((3, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'slr'
    offer['price'] = 0
    offer['time'] = 11 + EXPIRATION
    events = update(events, timefeats, thread_id=3, offer=offer, trigger_type=SLR_REJECTION)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((3, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'byr'
    offer['price'] = .6
    offer['time'] = 12 + (2 * EXPIRATION)
    events = update(events, timefeats, thread_id=1, offer=offer,
                    trigger_type=BYR_REJECTION)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((3, offer['time']), timefeats, exp, time_checks)

    offer['price'] = .4
    offer['time'] = 13 + (2 * EXPIRATION)
    events = update(events, timefeats, thread_id=2, offer=offer,
                    trigger_type=BYR_REJECTION)
    get_exp_feats((3, offer['time']), timefeats, exp, time_checks)

    offer['price'] = 1
    offer['time'] = 14 + (2 * EXPIRATION)
    events = update(events, timefeats, thread_id=3, offer=offer,
                    trigger_type=ACCEPTANCE)
    compare_all(events, exp, time_checks)


def test_partial_seq_partial_overlap_expire(timefeats):
    offer = {
        'type': 'byr',
        'price': .5,
        'time': 4,
    }
    time_checks = list()
    exp = list()
    events = update(None, timefeats, thread_id=1, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'slr'
    offer['price'] = .1
    offer['time'] = 5
    events = update(events, timefeats, thread_id=1, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'byr'
    offer['price'] = .6
    offer['time'] = 6
    events = update(events, timefeats, thread_id=1, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'slr'
    offer['price'] = .2
    offer['time'] = 7
    events = update(events, timefeats, thread_id=1, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'byr'
    offer['price'] = .4
    offer['time'] = 8 + EXPIRATION
    events = update(events, timefeats, thread_id=2, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'slr'
    offer['price'] = .2
    offer['time'] = 9 + EXPIRATION
    events = update(events, timefeats, thread_id=2, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'byr'
    offer['price'] = .65
    offer['time'] = 10 + EXPIRATION
    events = update(events, timefeats, thread_id=3, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((3, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'slr'
    offer['price'] = .2
    offer['time'] = 11 + EXPIRATION
    events = update(events, timefeats, thread_id=3, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((3, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'byr'
    offer['price'] = .6
    offer['time'] = 12 + EXPIRATION
    events = update(events, timefeats, thread_id=1, offer=offer,
                    trigger_type=BYR_REJECTION)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((3, offer['time']), timefeats, exp, time_checks)

    offer['price'] = .4
    offer['time'] = 13 + EXPIRATION
    events = update(events, timefeats, thread_id=2, offer=offer,
                    trigger_type=BYR_REJECTION)
    get_exp_feats((3, offer['time']), timefeats, exp, time_checks)

    offer['price'] = .8
    offer['time'] = 14 + EXPIRATION
    events = update(events, timefeats, thread_id=3, offer=offer,
                    trigger_type=ACCEPTANCE)
    compare_all(events, exp, time_checks)


def test_sequential_rej_byr_accept_expire(timefeats):
    offer = {
        'type': 'byr',
        'price': .5,
        'time': 4,
    }
    time_checks = list()
    exp = list()
    events = add_event(None, trigger_type=OFFER, thread_id=1, offer=offer)
    timefeats.update_features(trigger_type=OFFER, thread_id=1, offer=offer)
    get_exp_feats((1, 4), timefeats, exp, time_checks)
    offer['price'] = .2
    offer['time'] = 6
    offer['type'] = 'slr'
    events = add_event(events, trigger_type=OFFER, thread_id=1, offer=offer)
    timefeats.update_features(trigger_type=OFFER, thread_id=1, offer=offer)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'byr'
    offer['price'] = .63
    offer['time'] = 7
    events = add_event(events, trigger_type=OFFER, thread_id=1, offer=offer)
    timefeats.update_features(trigger_type=OFFER, thread_id=1, offer=offer)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    offer['type'] = 'slr'
    offer['price'] = .3
    offer['time'] = 8
    events = add_event(events, trigger_type=OFFER, thread_id=1, offer=offer)
    timefeats.update_features(trigger_type=OFFER, thread_id=1, offer=offer)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'byr'
    offer['price'] = .63
    offer['time'] = 9
    events = add_event(events, trigger_type=BYR_REJECTION, thread_id=1, offer=offer)
    timefeats.update_features(trigger_type=BYR_REJECTION, thread_id=1, offer=offer)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)

    offer['time'] = 10 + EXPIRATION
    offer['price'] = .6
    events = add_event(events, trigger_type=OFFER, thread_id=2, offer=offer)
    timefeats.update_features(trigger_type=OFFER, thread_id=2, offer=offer)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)

    offer['time'] = 11 + EXPIRATION
    offer['type'] = 'slr'
    offer['price'] = .3
    events = update(events, timefeats, thread_id=2, offer=offer, trigger_type=OFFER)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)

    # acceptance
    offer['price'] = .7
    offer['time'] = 12 + EXPIRATION
    offer['type'] = 'byr'
    events = add_event(events, trigger_type=ACCEPTANCE, thread_id=2, offer=offer)
    print('events input')
    print(events)
    compare_all(events, exp, time_checks)


def test_slr_accept_worst_buyer_expire(timefeats):
    offer = {
        'type': 'byr',
        'price': .5,
        'time': 4,
    }
    time_checks = list()
    exp = list()
    events = update(None, timefeats, thread_id=1, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, 4), timefeats, exp, time_checks)
    get_exp_feats((2, 4), timefeats, exp, time_checks)

    offer['type'] = 'byr'
    offer['price'] = .60
    offer['time'] = 5
    events = update(events, timefeats, thread_id=2, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, 5), timefeats, exp, time_checks)
    get_exp_feats((2, 5), timefeats, exp, time_checks)
    get_exp_feats((3, 5), timefeats, exp, time_checks)

    offer['type'] = 'byr'
    offer['price'] = .65
    offer['time'] = 6
    events = update(events, timefeats, thread_id=3, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((3, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'slr'
    offer['price'] = 0
    offer['time'] = 7 + EXPIRATION
    events = update(events, timefeats, thread_id=3, offer=offer, trigger_type=SLR_REJECTION)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((3, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'slr'
    offer['price'] = 0
    offer['time'] = 8 + EXPIRATION
    events = update(events, timefeats, thread_id=2, offer=offer, trigger_type=OFFER)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((3, offer['time']), timefeats, exp, time_checks)

    offer['type'] = 'slr'
    offer['price'] = .5
    offer['time'] = 9 + EXPIRATION
    events = update(events, timefeats, thread_id=1, offer=offer, trigger_type=ACCEPTANCE)
    compare_all(events, exp, time_checks)


def test_same_time_byr_offer_expire(timefeats):
    offer = {
        'type': 'byr',
        'price': .5,
        'time': 4,
    }
    time_checks = list()
    exp = list()
    events = update(None, timefeats, offer=offer, trigger_type=OFFER, thread_id=1)
    offer['price'] = .6
    events = update(events, timefeats, offer=offer, trigger_type=OFFER, thread_id=2)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)

    offer['time'] = 5 + EXPIRATION
    offer['type'] = 'slr'
    offer['price'] = 0
    events = update(events, timefeats, offer=offer, trigger_type=SLR_REJECTION, thread_id=1)
    get_exp_feats((1, offer['time']), timefeats, exp, time_checks)
    get_exp_feats((2, offer['time']), timefeats, exp, time_checks)

    offer['price'] = .4
    offer['time'] = 6 + EXPIRATION
    events = update(events, timefeats, offer=offer, trigger_type=ACCEPTANCE, thread_id=2)
    compare_all(events, exp, time_checks)
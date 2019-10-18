import pytest
import pandas as pd
import numpy as np
import torch
from processing.b_feats.tf_lstg import get_lstg_time_feats
from rlenv.TimeFeatures import TimeFeatures
from rlenv.time_triggers import *


# consts
COLS = ['accept', 'clock', 'reject', 'byr',
        'start_price', 'norm', 'price', 'censored', 'message']
ACCEPTANCE = 'acceptance' # time trigger for df not needed for environment


def compare_all(events, exp, time_checks):
    events = get_lstg_time_feats(events)
    act = [events.loc[(0, idx[0], idx[1])].values for idx in time_checks]
    for curr_act, curr_exp, idx in zip(act, exp, time_checks):
        print('')
        print('thread : {}, time: {}'.format(idx[0], idx[1]))
        compare(curr_act, curr_exp)


def compare(actual, exp):
    """
    Approximate equality between expected tensor and actual tensor

    :param actual: 1 dimensional torch.tensor
    :param exp: 1 dimensional torch.tensor
    :return: NA
    """
    if not isinstance(actual, torch.Tensor):
        actual = torch.from_numpy(actual).float()
    if not isinstance(exp, torch.Tensor):
        exp = torch.from_numpy(exp).float()
    assert torch.all(torch.lt(torch.abs(torch.add(actual, -exp)), 1e-6))


def get_exp_feats(idx, timefeats, exp, time_checks):
    new_feats = timefeats.get_feats(thread_id=idx[0], time=idx[1])
    exp.append(new_feats)
    time_checks.append(idx)


def update(events, timefeats, trigger_type=None, thread_id=None, offer=None):
    """
    Updates events DataFrame and TimeFeatures object with the same event
    :param events: pd.DataFrame containing events
    :param timefeats: instance of rlenv.TimeFeatures.TimeFeatures
    :param thread_id: int giving thread id
    :param offer: dictionary containing parameters of offer
    :param trigger_type: str defined in rlenv.time_triggers giving type of event
    :return: updated events df
    """
    events = add_event(events, trigger_type=trigger_type, thread_id=thread_id, offer=offer)
    if trigger_type == ACCEPTANCE:
        return events
    timefeats.update_features(trigger_type=trigger_type, thread_id=thread_id, offer=offer)
    return events


def add_event(df, trigger_type=None, offer=None, thread_id=None):
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
    if 0 in df.index.levels[0]:
        last_index = df.xs(0, level='lstg', drop_level=True)
        if thread_id in last_index.index.levels[0]:
            last_index = df.xs(thread_id, level='thread',
                               drop_level=True).reset_index()['index'].max() + 1
        else:
            last_index = 1
    else:
        last_index = 1
    offer_index = pd.MultiIndex.from_tuples([(0, thread_id, last_index)], names=['lstg', 'thread', 'index'])

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
    t1_o1 = timefeats.get_feats(thread_id=1, time=4)
    t2_o1 = timefeats.get_feats(thread_id=2, time=4)
    offer['price'] = .65
    offer['time'] = 6
    events = add_event(events, trigger_type=OFFER, thread_id=2, offer=offer)
    timefeats.update_features(trigger_type=OFFER, thread_id=2, offer=offer)
    t1_o2 = timefeats.get_feats(thread_id=1, time=6)
    t2_o2 = timefeats.get_feats(thread_id=2, time=6)
    offer['price'] = .9
    offer['time'] = 8
    offer['type'] = 'slr'
    events = add_event(events, trigger_type=OFFER, thread_id=2, offer=offer)
    timefeats.update_features(trigger_type=OFFER, thread_id=2, offer=offer)
    t1_o3 = timefeats.get_feats(thread_id=1, time=8)
    t2_o3 = timefeats.get_feats(thread_id=2, time=8)
    offer['price'] = .85
    offer['time'] = 9
    events = add_event(events, trigger_type=OFFER, thread_id=1, offer=offer)
    timefeats.update_features(trigger_type=OFFER, thread_id=1, offer=offer)
    t1_o4 = timefeats.get_feats(thread_id=1, time=9)
    t2_o4 = timefeats.get_feats(thread_id=2, time=9)
    # rejection
    offer['price'] = .65
    offer['time'] = 10
    offer['type'] = 'byr'
    events = add_event(events, trigger_type=BYR_REJECTION, thread_id=2, offer=offer)
    timefeats.update_features(trigger_type=BYR_REJECTION, thread_id=2)
    t1_o5 = timefeats.get_feats(thread_id=1, time=10)
    offer['time'] = 11
    offer['price'] = .5
    events = add_event(events, trigger_type=BYR_REJECTION, thread_id=1, offer=offer)
    print('')
    print(events)
    act = get_lstg_time_feats(events)
    print(act.loc[:, ['byr_offers_open', 'byr_best_open', 'byr_best', 'byr_offers']])
    print(act.loc[:, ['slr_offers_open', 'slr_best_open', 'slr_best', 'slr_offers']])
    compare(act.loc[(0, 1, 4)].values, t1_o1)
    compare(act.loc[(0, 2, 4)].values, t2_o1)
    compare(act.loc[(0, 1, 6)].values, t1_o2)
    compare(act.loc[(0, 2, 6)].values, t2_o2)
    compare(act.loc[(0, 2, 8)].values, t2_o3)
    compare(act.loc[(0, 1, 8)].values, t1_o3)
    compare(act.loc[(0, 2, 9)].values, t2_o4)
    compare(act.loc[(0, 1, 9)].values, t1_o4)
    compare(act.loc[(0, 1, 10)].values, t1_o5)


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




import pytest
from processing.c_feats.test.test_utils import (compare_all,
                                                update, add_event,
                                                get_exp_feats)
from rlenv.time.TimeFeatures import TimeFeatures
from rlenv.time.Offer import Offer
from rlenv.env_consts import BYR_PREFIX, SLR_PREFIX


@pytest.fixture()
def timefeats():
    return TimeFeatures()


def test_censoring_tf(timefeats):
    # thread 1
    t1o1 = Offer(params={'price': .54,
                         'time': 15,
                         'player': BYR_PREFIX,
                         'thread_id': 1})
    t1o2 = Offer(params={'price': .152,
                         'time': 90,
                         'player': SLR_PREFIX,
                         'thread_id': 1})
    t1o3 = Offer(params={'price': 1,
                         'time': 120,
                         'censored': True,
                         'player': BYR_PREFIX,
                         'thread_id': 1}, rej=True)
    # second thread
    t2o1 = Offer(params={'price': .65,
                         'time': 100,
                         'player': BYR_PREFIX,
                         'thread_id': 2})
    t2o2 = Offer(params={'price': .154,
                         'time': 110,
                         'player': SLR_PREFIX,
                         'thread_id': 2})
    t2o3 = Offer(params={'price': 1-.154,
                         'time': 120,
                         'player': BYR_PREFIX,
                         'thread_id': 2}, accept=True)
    exp, time_checks = list(), list()
    events = update(events=None, timefeats=timefeats, offer=t1o1)
    get_exp_feats((1, 15), timefeats, exp, time_checks)
    get_exp_feats((2, 15), timefeats, exp, time_checks)

    events = update(events=events, timefeats=timefeats, offer=t1o2)
    get_exp_feats((1, 90), timefeats, exp, time_checks)
    get_exp_feats((2, 90), timefeats, exp, time_checks)

    events = update(events=events, timefeats=timefeats, offer=t2o1)
    get_exp_feats((1, 100), timefeats, exp, time_checks)
    get_exp_feats((2, 100), timefeats, exp, time_checks)

    events = update(events=events, timefeats=timefeats, offer=t2o2)
    get_exp_feats((1, 110), timefeats, exp, time_checks)
    get_exp_feats((2, 110), timefeats, exp, time_checks)

    events = add_event(events, offer=t2o3)
    get_exp_feats((2, 120), timefeats, exp, time_checks)
    events = update(events=events, timefeats=timefeats, offer=t1o3)
    compare_all(events=events, exp=exp, time_checks=time_checks, lstg=1)



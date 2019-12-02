from constants import BYR_PREFIX, SLR_PREFIX
from rlenv.env_utils import *
import rlenv.interface.model_names as model_names
import pytest


def test_model_collections():
    assert len(model_names.ARRIVAL) == 2
    assert model_names.BYR_HIST in model_names.ARRIVAL
    assert model_names.NUM_OFFERS in model_names.ARRIVAL
    assert len(model_names.OFFER_NO_PREFIXES) == 3
    assert len(model_names.OFFER) == 6
    assert len([model for model in model_names.OFFER if model_names.MSG in model]) == 2
    assert len([model for model in model_names.OFFER if model_names.DELAY in model]) == 2
    assert len([model for model in model_names.OFFER if model_names.CON in model]) == 2
    assert len([model for model in model_names.OFFER if SLR_PREFIX in model]) == 3
    assert len([model for model in model_names.OFFER if BYR_PREFIX in model]) == 3
    assert len(model_names.FEED_FORWARD) == 5
    assert len([model for model in model_names.FEED_FORWARD if model_names.CON in model]) == 2
    assert len([model for model in model_names.FEED_FORWARD if model_names.MSG in model]) == 2
    assert len([model for model in model_names.FEED_FORWARD if SLR_PREFIX in model]) == 2
    assert len([model for model in model_names.FEED_FORWARD if BYR_PREFIX in model]) == 2
    assert model_names.BYR_HIST in model_names.FEED_FORWARD
    assert len(model_names.RECURRENT) == 3
    assert len([model for model in model_names.RECURRENT if SLR_PREFIX in model]) == 1
    assert len([model for model in model_names.RECURRENT if BYR_PREFIX in model]) == 1
    assert model_names.NUM_OFFERS in model_names.RECURRENT
    assert len([model for model in model_names.RECURRENT if model_names.DELAY in model]) == 2
    assert len([model for model in model_names.LSTM_MODELS if SLR_PREFIX in model]) == 1
    assert len([model for model in model_names.LSTM_MODELS if BYR_PREFIX in model]) == 1
    assert len([model for model in model_names.LSTM_MODELS if model_names.DELAY in model]) == 2
    assert model_names.NUM_OFFERS in model_names.LSTM_MODELS


def test_load_featnames_sizes():
    for model in model_names.MODELS:
        featnames = load_featnames(model)
        assert 'x_fixed' in featnames
        if model in model_names.RECURRENT:
            assert 'x_time' in featnames
        sizes = load_sizes(model)
        assert 'fixed' in sizes
        assert 'out' in sizes
        assert len(featnames['x_fixed']) == sizes['fixed']
        if model in model_names.RECURRENT:
            assert 'time' in sizes
            assert len(featnames['x_time']) == sizes['time']


def test_get_model_class():
    for name in model_names.MODELS:
        curr_class = get_model_class(name)
        if name == 'delay_byr' or name == 'delay_slr' or name == 'arrival':
            assert curr_class == LSTM
        else:
            assert curr_class == FeedForward


def test_get_clock_feats():
    christmas_morn = 17884800
    a = get_clock_feats(christmas_morn)
    assert a[0] == 1
    day_before = christmas_morn - 24 * 60 * 60
    a = get_clock_feats(day_before)
    assert a[0] == 0
    for i in range(7):
        curr_day = day_before + i * 24 * 60 * 60
        a = get_clock_feats(curr_day)
        if i < 6:
            assert a[i + 1] == 1
            assert a[1:7].sum() == 1
        else:
            assert a[1:7].sum() == 0
    assert a[7] == 0
    curr_day = day_before + 12 * 60 * 60
    a = get_clock_feats(curr_day)
    assert a[7] == 0.5
    curr_day = curr_day + 30 * 60
    a = get_clock_feats(curr_day)
    assert a[7] == (.5 + 30 * 60 / DAY)
    curr_day = curr_day + 30
    a = get_clock_feats(curr_day)
    assert a[7] == (.5 + 30 * 60 / DAY + 30 / DAY)


def test_proper_squeeze():
    two = torch.zeros(5, 1)
    assert len(proper_squeeze(two).shape) == 1
    three = torch.zeros(1, 5, 1)
    assert len(proper_squeeze(three).shape) == 1
    one = torch.zeros(5)
    assert len(proper_squeeze(one).shape) == 1
    singleton = torch.zeros(1)
    assert len(proper_squeeze(singleton).shape) == 1
    double = torch.zeros(5, 2)
    assert len(proper_squeeze(double).shape) == 2


def test_categorical_sample():
    params = torch.ones(1, 5).float()
    samp = categorical_sample(params, 1)
    assert len(samp.shape) == 1
    params = torch.ones(1, 3).float()
    outcomes = dict()
    outcomes[0] = 0
    outcomes[1] = 0
    outcomes[2] = 0
    for _ in range(10000):
        samp = categorical_sample(params, 1)
        samp = int(samp.item())
        outcomes[samp] += 1
    for i in outcomes:
        print('{}: {}'.format(i, outcomes[i]))
    for i in outcomes:
        assert outcomes[i] > 1000
        assert outcomes[i] < 5000
    assert len(outcomes) == 3


def test_get_split():
    a = torch.tensor([.49]).float()
    b = torch.tensor([.48]).float()
    c = torch.tensor([.51]).float()
    d = torch.tensor([.52]).float()
    e = torch.tensor([.5]).float()
    f = torch.tensor([.2]).float()
    assert get_split(a) == 1
    assert get_split(b) == 0
    assert get_split(c) == 1
    assert get_split(d) == 0
    assert get_split(e) == 1
    assert get_split(f) == 0


def test_load_model():
    for model in model_names.MODELS:
        load_model(model, 1)


def test_time_delta():
    assert time_delta(15, 15 + DAY, unit=DAY) == 1
    assert time_delta(15, 15 + MONTH, unit=MONTH) == 1
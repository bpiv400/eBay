from rlenv.env_utils import *
import rlenv.env_consts as model_names
from model.nets import FeedForward
import pytest


def test_model_collections():
    assert len(model_names.ARRIVAL) == 2
    assert model_names.BYR_HIST_MODEL in model_names.ARRIVAL
    assert model_names.ARRIVAL_MODEL in model_names.ARRIVAL
    assert len(model_names.OFFER_NO_PREFIXES) == 3
    assert len(model_names.OFFER_MODELS) == 6
    assert len([model for model in model_names.OFFER_MODELS if model_names.MSG in model]) == 2
    assert len([model for model in model_names.OFFER_MODELS if model_names.DELAY in model]) == 2
    assert len([model for model in model_names.OFFER_MODELS if model_names.CON in model]) == 2
    assert len([model for model in model_names.OFFER_MODELS if SLR_PREFIX in model]) == 3
    assert len([model for model in model_names.OFFER_MODELS if BYR_PREFIX in model]) == 3
    assert len(model_names.FEED_FORWARD_MODELS) == 5
    assert len([model for model in model_names.FEED_FORWARD_MODELS if model_names.CON in model]) == 2
    assert len([model for model in model_names.FEED_FORWARD_MODELS if model_names.MSG in model]) == 2
    assert len([model for model in model_names.FEED_FORWARD_MODELS if SLR_PREFIX in model]) == 2
    assert len([model for model in model_names.FEED_FORWARD_MODELS if BYR_PREFIX in model]) == 2
    assert model_names.BYR_HIST_MODEL in model_names.FEED_FORWARD_MODELS
    assert len(model_names.RECURRENT_MODELS) == 3
    assert len([model for model in model_names.RECURRENT_MODELS if SLR_PREFIX in model]) == 1
    assert len([model for model in model_names.RECURRENT_MODELS if BYR_PREFIX in model]) == 1
    assert model_names.ARRIVAL_MODEL in model_names.RECURRENT_MODELS
    assert len([model for model in model_names.RECURRENT_MODELS if model_names.DELAY in model]) == 2
    assert len([model for model in model_names.LSTM_MODELS if SLR_PREFIX in model]) == 1
    assert len([model for model in model_names.LSTM_MODELS if BYR_PREFIX in model]) == 1
    assert len([model for model in model_names.LSTM_MODELS if model_names.DELAY in model]) == 2
    assert model_names.ARRIVAL_MODEL in model_names.LSTM_MODELS


def test_get_model_class():
    for name in model_names.MODELS:
        curr_class = get_model_class(name)
        if name == 'delay_byr' or name == 'delay_slr' or name == 'arrival':
            assert curr_class == Recurrent
        else:
            assert curr_class == FeedForward


def test_get_clock_feats():
    christmas_morn = 17884800
    a = get_clock_feats(christmas_morn)
    assert a[0] == 1
    day_before = christmas_morn - 24 * 60 * 60
    a = get_clock_feats(day_before)
    assert a[0] == pytest.approx(0)
    for i in range(7):
        curr_day = day_before + i * 24 * 60 * 60
        a = get_clock_feats(curr_day)
        if i < 6:
            assert a[i + 1] == pytest.approx(1)
            assert a[1:7].sum() == pytest.approx(1)
        else:
            assert a[1:7].sum() == pytest.approx(0)
    assert a[7] == pytest.approx(0)
    curr_day = day_before + 12 * 60 * 60
    a = get_clock_feats(curr_day)
    assert a[7] == pytest.approx(0.5)
    curr_day = curr_day + 30 * 60
    a = get_clock_feats(curr_day)
    assert a[7] == pytest.approx(.5 + 30 * 60 / DAY)
    curr_day = curr_day + 30
    a = get_clock_feats(curr_day)
    assert a[7] == pytest.approx(.5 + 30 * 60 / DAY + 30 / DAY)


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
    samp = sample_categorical(params, 1)
    assert len(samp.shape) == 1
    params = torch.ones(1, 3).float()
    outcomes = dict()
    outcomes[0] = 0
    outcomes[1] = 0
    outcomes[2] = 0
    for _ in range(10000):
        samp = sample_categorical(params, 1)
        samp = int(samp)
        outcomes[samp] += 1
    for i in outcomes:
        print('{}: {}'.format(i, outcomes[i]))
    for i in outcomes:
        assert outcomes[i] > 1000
        assert outcomes[i] < 5000
    assert len(outcomes) == 3


def test_get_split():
    a = torch.tensor([.49]).numpy()
    b = torch.tensor([.48]).numpy()
    c = torch.tensor([.51]).numpy()
    d = torch.tensor([.52]).numpy()
    e = torch.tensor([.5]).numpy()
    f = torch.tensor([.2]).numpy()
    assert get_split(a) == 1
    assert get_split(b) == 0
    assert get_split(c) == 1
    assert get_split(d) == 0
    assert get_split(e) == 1
    assert get_split(f) == 0


def test_load_model():
    for model in model_names.MODELS:
        print(model)
        load_model(model)
from math import exp as exp
import pytest
import torch
import pandas as pd
from rlenv.SimulatorInterface import SimulatorInterface
from rlenv.env_consts import EXPERIMENT_PATH, MODEL_DIR
from rlenv.model_names import (MODELS, OFFER_NO_PREFIXES,
                               ARRIVAL, ARRIVAL_PREFIX, OFFER, DELAY, DAYS, FEED_FORWARD)
from constants import SLR_PREFIX, BYR_PREFIX
from simulator.nets import LSTM, RNN, FeedForward

@pytest.fixture
def simulator():
    params = pd.read_csv(EXPERIMENT_PATH)
    params.set_index('id', drop=True, inplace=True)
    params = params.loc[1, :].to_dict()
    simulator = SimulatorInterface(params)


def test_model_dirs():
    targs = ['{}/{}/{}/'.format(MODEL_DIR, SLR_PREFIX, offer) for offer in OFFER_NO_PREFIXES]
    targs = targs + ['{}/{}/{}/'.format(MODEL_DIR, BYR_PREFIX, offer) for offer in OFFER_NO_PREFIXES]
    targs = targs + ['{}/{}/{}/'.format(MODEL_DIR, ARRIVAL_PREFIX, offer) for offer in ARRIVAL]
    actuals = [SimulatorInterface._model_dir(model) for model in MODELS]
    assert all([targ == actual for targ, actual in zip(targs, actuals)])


def test_model_types():
    for model in OFFER:
        if DELAY in model:
            assert SimulatorInterface._model_type(model) is LSTM
        else:
            assert SimulatorInterface._model_type(model) is RNN
    assert SimulatorInterface._model_type(DAYS) is LSTM
    assert all([SimulatorInterface._model_type(model_name) is FeedForward for model_name in FEED_FORWARD])


def test_init():
    params = pd.read_csv(EXPERIMENT_PATH)
    params.set_index('id', drop=True, inplace=True)
    params = params.loc[1, :].to_dict()
    simulator = SimulatorInterface(params)


def test_binary_dist_shape_single():
    params = torch.tensor([[2]]).float()
    assert list(params.shape) == [1, 1]
    assert list(SimulatorInterface._bernoulli_sample(params, 1).shape) == [1]


def test_binary_dist_shape_single_mult_samp():
    params = torch.tensor([[2]]).float()
    assert list(params.shape) == [1, 1]
    assert list(SimulatorInterface._bernoulli_sample(params, 4).shape) == [4]


def test_binary_dist_shape_mult():
    params = torch.tensor([[2], [-1], [1]]).float()
    assert list(params.shape) == [3, 1]
    assert list(SimulatorInterface._bernoulli_sample(params, 1).shape) == [3]


def test_beta_prep_singleton_sample_singleton_mixture():
    params = torch.tensor([[1.2, 1.3, 2]]).float()
    trans = SimulatorInterface._beta_prep(params)
    expect = torch.tensor([[[exp(1.2) + 1, exp(1.3) + 1, 2]]]).float()
    assert list(trans.shape) == list(expect.shape)
    assert torch.all(torch.lt(torch.abs(torch.add(expect, -trans)), 1e-6))


def test_beta_prep_singleton_sample_multiple_mixture():
    params = torch.tensor([[1.2, 1.6, 1.3, 1.9, 2, 3]]).float()
    trans = SimulatorInterface._beta_prep(params)
    expect = torch.tensor([[[exp(1.2) + 1, exp(1.3) + 1, 2],
                            [exp(1.6) + 1, exp(1.9) + 1, 3]]]).float()
    assert list(trans.shape) == list(expect.shape)
    assert torch.all(torch.lt(torch.abs(torch.add(expect, -trans)), 1e-6))


def test_beta_prep_multiple_sample_singleton_mixture():
    params = torch.tensor([[1.2, 1.6, 1.3],
                           [2.3, 1.1, -4]]).float()
    trans = SimulatorInterface._beta_prep(params)
    expect = torch.tensor([[[exp(1.2) + 1, exp(1.6) + 1, 1.3]],
                           [[exp(2.3) + 1, exp(1.1) + 1, -4]]]).float()
    assert list(trans.shape) == list(expect.shape)
    print(expect)
    print(trans)
    assert torch.all(torch.lt(torch.abs(torch.add(expect, -trans)), 1e-6))


def test_beta_prep_multiple_sample_multiple_mixture():
    params = torch.tensor([[1.2, 1.6, 1.3, 1.9, 2, 3],
                           [2.3, 1.1, 1.1, 1.05, 3, -3]]).float()
    trans = SimulatorInterface._beta_prep(params)
    expect = torch.tensor([[[exp(1.2) + 1, exp(1.3) + 1, 2],
                           [exp(1.6) + 1, exp(1.9) + 1, 3]],
                           [[exp(2.3) + 1, exp(1.1) + 1, 3],
                            [exp(1.1) + 1, exp(1.05) + 1, -3]]]).float()
    first_samp = torch.tensor([[exp(1.2) + 1, exp(1.3) + 1, 2],
                              [exp(1.6) + 1, exp(1.9) + 1, 3]]).float()
    assert list(trans.shape) == list(expect.shape)
    assert torch.all(torch.lt(torch.abs(torch.add(expect, -trans)), 1e-6))
    assert torch.all(torch.lt(torch.abs(torch.add(expect[0, :, :], -first_samp)), 1e-6))


def test_beta_cat_singleton_sample_single_mixture():
    params = torch.tensor([[1.2, 1.3, 2]]).float()
    trans = SimulatorInterface._beta_prep(params)
    cat = SimulatorInterface._beta_ancestor(trans)
    print('logits: {}'.format(cat.logits))
from math import exp as exp
import pytest
import torch
import torch.nn.functional as F
import pandas as pd
from rlenv.SimulatorInterface import SimulatorInterface
from rlenv.env_consts import EXPERIMENT_PATH, MODEL_DIR
from interface.model_names import (MODELS, OFFER_NO_PREFIXES,
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
    assert list(cat.batch_shape) == [1]
    exp = torch.tensor([[1]]).float()
    assert cat._num_events == 1
    assert torch.all(torch.lt(torch.abs(torch.add(exp, -cat.probs)), 1e-6))


def test_beta_cat_singleton_sample_multiple_mixture():
    params = torch.tensor([[1.2, 1.6, 1.3, 1.9, 2, 3]]).float()
    trans = SimulatorInterface._beta_prep(params)
    cat = SimulatorInterface._beta_ancestor(trans)
    exp = F.softmax(torch.tensor([[2, 3]]).float(), dim=-1)
    assert list(cat.batch_shape) == [1]
    assert cat._num_events == 2
    assert torch.all(torch.lt(torch.abs(torch.add(exp, -cat.probs)), 1e-6))


def test_beta_cat_multiple_sample_single_mixture():
    params = torch.tensor([[1.2, 1.6, 1.3],
                          [2.3, 1.1, -4]]).float()
    trans = SimulatorInterface._beta_prep(params)
    cat = SimulatorInterface._beta_ancestor(trans)
    exp = torch.tensor([[1], [1]]).float()
    assert list(cat.probs.shape) == [2, 1]
    assert list(cat.batch_shape) == [2]
    assert cat._num_events == 1
    assert torch.all(torch.lt(torch.abs(torch.add(exp, -cat.probs)), 1e-6))


def test_beta_cat_multiple_sample_multiple_mixture():
    params = torch.tensor([[1.2, 1.6, 1.3, 1.9, 2, 3],
                           [2.3, 1.1, 1.1, 1.05, 3, -3]]).float()
    trans = SimulatorInterface._beta_prep(params)
    cat = SimulatorInterface._beta_ancestor(trans)
    exp = torch.tensor([[2, 3], [3, -3]]).float()
    exp = F.softmax(exp, dim=-1)
    assert list(cat.probs.shape) == [2, 2]
    assert list(cat.batch_shape) == [2]
    assert cat._num_events == 2
    assert torch.all(torch.lt(torch.abs(torch.add(exp, -cat.probs)), 1e-6))


def test_beta_cat_more_samples_than_mixtures():
    params = torch.tensor([[1.2, 1.6, 1.3, 1.9, 2, 3],
                           [2.3, 1.1, 1.1, 1.05, 3, -3],
                           [1.4, 1.5, 1.9, 1.3, 4, 4.1]]).float()
    trans = SimulatorInterface._beta_prep(params)
    cat = SimulatorInterface._beta_ancestor(trans)
    exp = torch.tensor([[2, 3], [3, -3], [4, 4.1]]).float()
    exp = F.softmax(exp, dim=-1)
    assert list(cat.probs.shape) == [3, 2]
    assert list(cat.batch_shape) == [3]
    assert cat._num_events == 2
    assert torch.all(torch.lt(torch.abs(torch.add(exp, -cat.probs)), 1e-6))


def test_beta_cat_more_mixtures_than_samples():
    params = torch.tensor([[1.2, 1.6, 1.7, 1.3, 1.9, 2, 2, 3, 2.5],
                           [2.3, 1.1, 1.9, 1.1, 1.05, 1.6, 3, -3, 2]]).float()
    trans = SimulatorInterface._beta_prep(params)
    cat = SimulatorInterface._beta_ancestor(trans)
    exp = torch.tensor([[2, 3, 2.5], [3, -3, 2]]).float()
    exp = F.softmax(exp, dim=-1)
    assert list(cat.probs.shape) == [2, 3]
    assert list(cat.batch_shape) == [2]
    assert cat._num_events == 3
    assert torch.all(torch.lt(torch.abs(torch.add(exp, -cat.probs)), 1e-6))


def test_beta_cat_singleton_sample_single_mixture_sample():
    params = torch.tensor([[1.2, 1.3, 2]]).float()
    trans = SimulatorInterface._beta_prep(params)
    cat = SimulatorInterface._beta_ancestor(trans)
    draws = cat.sample(sample_shape=(1,))
    assert list(draws.shape) == [1, 1]


def test_beta_cat_singleton_sample_multiple_mixture_sample():
    params = torch.tensor([[1.2, 1.6, 1.3, 1.9, 2, 3]]).float()
    trans = SimulatorInterface._beta_prep(params)
    cat = SimulatorInterface._beta_ancestor(trans)
    draws = cat.sample(sample_shape=(1,))
    assert list(draws.shape) == [1, 1]


def test_beta_cat_multiple_sample_single_mixture_sample():
    params = torch.tensor([[1.2, 1.6, 1.3],
                          [2.3, 1.1, -4]]).float()
    trans = SimulatorInterface._beta_prep(params)
    cat = SimulatorInterface._beta_ancestor(trans)
    exp = torch.tensor([[1], [1]]).float()
    draws = cat.sample(sample_shape=(1,))
    assert list(draws.shape) == [1, 2]


def test_beta_cat_multiple_sample_multiple_mixture_sample():
    params = torch.tensor([[1.2, 1.6, 1.3, 1.9, 2, 3],
                           [2.3, 1.1, 1.1, 1.05, 3, -3]]).float()
    trans = SimulatorInterface._beta_prep(params)
    cat = SimulatorInterface._beta_ancestor(trans)
    draws = cat.sample(sample_shape=(1,))
    assert list(draws.shape) == [1, 2]


def test_make_beta_params_singleton_sample_single_mixture():
    params = torch.tensor([[1.2, 1.3, 2]]).float()
    trans = SimulatorInterface._beta_prep(params)
    ancestor = SimulatorInterface._beta_ancestor(trans)
    beta_params = SimulatorInterface._make_beta_params(ancestor, trans)
    expect = torch.tensor([[exp(1.2) + 1, exp(1.3) + 1]]).float()
    assert list(expect.shape) == list(beta_params.shape)
    assert torch.all(torch.lt(torch.abs(torch.add(expect, -beta_params)), 1e-6))


def test_make_beta_params_singleton_sample_multiple_mixture():
    params = torch.tensor([[1.2, 1.6, 1.3, 1.9, 2, 3]]).float()
    trans = SimulatorInterface._beta_prep(params)
    ancestor = SimulatorInterface._beta_ancestor(trans)
    beta_params = SimulatorInterface._make_beta_params(ancestor, trans)
    expect_1 = torch.tensor([[exp(1.2) + 1, exp(1.3) + 1]]).float()
    expect_2 = torch.tensor([[exp(1.6) + 1, exp(1.9) + 1]]).float()
    assert list(expect_1.shape) == list(beta_params.shape)
    first = torch.all(torch.lt(torch.abs(torch.add(expect_1, -beta_params)), 1e-6))
    second = torch.all(torch.lt(torch.abs(torch.add(expect_2, -beta_params)), 1e-6))
    assert first or second


def test_make_beta_params_multiple_sample_single_mixture():
    params = torch.tensor([[1.2, 1.6, 1.3],
                           [2.3, 1.1, -4]]).float()
    trans = SimulatorInterface._beta_prep(params)
    ancestor = SimulatorInterface._beta_ancestor(trans)
    beta_params = SimulatorInterface._make_beta_params(ancestor, trans)
    expect_1 = torch.tensor([[exp(1.2) + 1, exp(1.6) + 1],
                             [exp(2.3) + 1, exp(1.1) + 1]]).float()
    print(beta_params)
    assert list(expect_1.shape) == list(beta_params.shape)
    assert torch.all(torch.lt(torch.abs(torch.add(expect_1, -beta_params)), 1e-6))







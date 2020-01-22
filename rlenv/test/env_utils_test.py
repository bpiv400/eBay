from rlenv.env_utils import *
import rlenv.env_consts as model_names
from constants import ENV_SIM_DIR
from rlenv.Composer import Composer


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


def test_create_composer():
    d = load('{}{}/chunks/{}.gz'.format(ENV_SIM_DIR, 'train_rl', 1))
    x_lstg = d['x_lstg']
    Composer(x_lstg.columns)
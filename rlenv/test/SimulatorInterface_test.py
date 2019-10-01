import pytest
import pandas as pd
from rlenv.SimulatorInterface import SimulatorInterface
from rlenv.env_consts import EXPERIMENT_PATH


@pytest.fixture
def simulator():
    params = pd.read_csv(EXPERIMENT_PATH)
    params.set_index('id', drop=True, inplace=True)
    params = params.loc[1, :].to_dict()
    simulator = SimulatorInterface(params)


def test_init():
    params = pd.read_csv(EXPERIMENT_PATH)
    params.set_index('id', drop=True, inplace=True)
    params = params.loc[1, :].to_dict()
    simulator = SimulatorInterface(params)

def test_binary_dist():
    pass

def test_binary_dist_shape_single():
    pass

def test_binary_dist_shape_mult():
    pass

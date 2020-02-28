import numpy as np
from constants import REINFORCE_DIR
from agent.agent_consts import INPUT_DIR, FULL_CON, QUARTILES, HALF


def get_con_set(con):
    if con == FULL_CON:
        return np.linspace(0, 100, 101) / 100
    elif con == QUARTILES:
        return np.array([0, 0.25, 0.50, 0.75, 1.0])
    elif con == HALF:
        return np.array([0.0, 0.50, 1.0])
    else:
        raise RuntimeError("Invalid concession set type parameter")


def slr_input_path(part=None):
    return '{}/{}/{}/slr.hdf5'.format(REINFORCE_DIR, part, INPUT_DIR)

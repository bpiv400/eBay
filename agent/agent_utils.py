from constants import REINFORCE_DIR
from agent.agent_consts import INPUT_DIR


def slr_input_path(part=None):
    return '{}/{}/{}/slr.hdf5'.format(REINFORCE_DIR, part, INPUT_DIR)

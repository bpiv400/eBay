import numpy as np
import torch
from agent.util import get_turn
from rlenv.const import DAYS_IND, NORM_IND
from agent.const import AGENT_CONS
from constants import NUM_COMMON_CONS, MAX_DAYS
from featnames import LSTG


def get_elapsed(x=None, turn=None):
    """
    Fraction of listing window elapsed.
    :param dict x: input features
    :param int turn: one of [2, 4, 6]
    :return: float
    """
    elapsed = x[LSTG][-5].item() / MAX_DAYS
    for i in range(2, turn + 1):
        elapsed += x['offer{}'.format(i)][DAYS_IND].item() / MAX_DAYS
    assert elapsed < 1.
    return elapsed


class HeuristicSlr:
    def __init__(self, delta=None):
        self.delta = delta

    def __call__(self, observation=None):
        # noinspection PyProtectedMember
        x = observation._asdict()

        # turn number
        turn_feats = x[LSTG][-2:].unsqueeze(dim=0)
        turn = int(get_turn(x=turn_feats, byr=False).item())
        assert turn in [2, 4, 6]

        # index of action
        cons = AGENT_CONS[turn]
        if turn == 2:
            elapsed = get_elapsed(x=x, turn=turn)
            if elapsed <= .61:
                idx = np.nonzero(cons == 0)[0][0]
            else:
                idx = np.nonzero(cons == 1)[0][0]

        elif turn == 4:
            elapsed = get_elapsed(x=x, turn=turn)
            if elapsed <= .26:
                idx = np.nonzero(cons == 0)[0][0]
            else:
                idx = np.nonzero(cons == 1)[0][0]

        elif turn == 6:
            # norm = x['offer{}'.format(turn - 1)][NORM_IND].item()
            idx = np.nonzero(cons == 1)[0][0]

        else:
            raise ValueError('Invalid turn: {}'.format(turn))

        # deterministic categorical action distribution
        pdf = torch.zeros(NUM_COMMON_CONS + 3, dtype=torch.float)
        pdf[idx] = 1.
        return pdf

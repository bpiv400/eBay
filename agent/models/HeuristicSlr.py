import numpy as np
import torch
from agent.models.util import wrapper, get_elapsed, get_last_norm, get_agent_turn
from agent.const import DELTA_SLR
from constants import NUM_COMMON_CONS
from featnames import DELTA


class HeuristicSlr:
    def __init__(self, **params):
        self.high = np.isclose(params[DELTA], DELTA_SLR[-1])

    def __call__(self, observation=None):
        # noinspection PyProtectedMember
        x = observation._asdict()

        # turn number
        turn = get_agent_turn(x=x, byr=False)

        # index of action
        f = wrapper(turn)
        if turn == 2:
            elapsed = get_elapsed(x=x, turn=turn)
            norm = get_last_norm(turn=turn, x=x)
            if self.high:
                if elapsed <= .49:
                    idx = f(0)
                else:
                    idx = f(.67) if norm <= .51 else f(1)
            else:
                if elapsed <= .38:
                    idx = f(0) if norm > .62 else f(.67)
                else:
                    idx = f(1)

        elif turn == 4:
            norm = get_last_norm(turn=turn, x=x)
            if self.high:
                elapsed = get_elapsed(x=x, turn=turn)
                if elapsed <= .33:
                    idx = f(0)
                else:
                    idx = f(.67) if norm <= .67 else f(1)
            else:
                idx = f(.5) if norm <= .69 else f(1)

        elif turn == 6:
            if self.high:
                idx = f(1)
            else:
                norm = get_last_norm(turn=turn, x=x)
                if norm > .65:
                    idx = f(1)
                else:
                    elapsed = get_elapsed(x=x, turn=turn)
                    idx = f(.5) if elapsed <= .25 else f(1)
        else:
            raise ValueError('Invalid turn: {}'.format(turn))

        # deterministic categorical action distribution
        pdf = torch.zeros(NUM_COMMON_CONS + 3, dtype=torch.float)
        pdf[idx] = 1.
        return pdf

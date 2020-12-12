import numpy as np
import torch
from agent.models.util import wrapper, get_elapsed, get_last_norm, get_agent_turn
from agent.const import DELTA_SLR
from constants import NUM_COMMON_CONS
from featnames import DELTA, TURN_COST


class HeuristicSlr:
    def __init__(self, **params):
        self.high = np.isclose(params[DELTA], DELTA_SLR[-1])
        self.turn_cost = params[TURN_COST]

    def __call__(self, observation=None):
        # noinspection PyProtectedMember
        x = observation._asdict()

        # turn number
        turn = get_agent_turn(x=x, byr=False)

        # index of action
        f = wrapper(turn)
        if turn == 2:
            elapsed = get_elapsed(x=x, turn=turn)
            threshold = .61 if self.high else .38
            idx = f(0) if elapsed <= threshold else f(1)

        elif turn == 4:
            if self.high:
                elapsed = get_elapsed(x=x, turn=turn)
                idx = f(0) if elapsed <= .26 else f(1)
            else:
                norm = get_last_norm(turn=turn, x=x)
                idx = f(.5) if norm <= .69 else f(1)

        elif turn == 6:
            if self.high:
                idx = f(1)
            else:
                norm = get_last_norm(turn=turn, x=x)
                idx = f(.5) if norm <= .61 else f(1)
        else:
            raise ValueError('Invalid turn: {}'.format(turn))

        # deterministic categorical action distribution
        pdf = torch.zeros(NUM_COMMON_CONS + 3, dtype=torch.float)
        pdf[idx] = 1.
        return pdf

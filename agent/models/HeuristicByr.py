import numpy as np
import torch
from agent.models.util import wrapper, get_store, get_last_norm, \
    get_last_auto, get_agent_turn
from agent.const import DELTA_BYR
from constants import NUM_COMMON_CONS


class HeuristicByr:
    def __init__(self, delta=None, turn_cost=None):
        self.high = np.isclose(delta, DELTA_BYR[-1])
        self.turn_cost = turn_cost

    def __call__(self, observation=None):
        # noinspection PyProtectedMember
        x = observation._asdict()

        # turn number
        turn = get_agent_turn(x=x, byr=True)

        # index of action
        f = wrapper(turn)
        if turn == 1:
            if self.high:
                idx = f(.5)
            else:
                store = get_store(x)
                idx = f(0) if store else f(.5)

        elif turn == 3:
            if not self.high:
                idx = f(.17)
            elif self.turn_cost == 0:
                auto = get_last_auto(x=x, turn=turn)
                idx = f(.4) if auto else f(.17)
            else:
                idx = f(.4)

        elif turn == 5:
            idx = f(.4) if self.high else f(.17)

        elif turn == 7:
            if not self.high:
                idx = f(0)
            else:
                norm = get_last_norm(turn=turn, x=x)
                idx = f(1) if norm <= .91 else f(0)
        else:
            raise ValueError('Invalid turn: {}'.format(turn))

        # deterministic categorical action distribution
        pdf = torch.zeros(NUM_COMMON_CONS + 2, dtype=torch.float)
        pdf[idx] = 1.
        return pdf

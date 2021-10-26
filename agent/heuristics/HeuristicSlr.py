import numpy as np
import torch
from agent.heuristics.util import get_agent_turn, wrapper, get_days, \
    get_recent_byr_offers, get_last_norm
from agent.const import DELTA_SLR, NUM_COMMON_CONS


class HeuristicSlr:
    def __init__(self, delta=None):
        self.patient = np.isclose(delta, DELTA_SLR[-1])

    def __call__(self, observation=None):
        # noinspection PyProtectedMember
        x = observation._asdict()

        # turn number
        turn = get_agent_turn(x=x, byr=False)

        # index of action
        f = wrapper(turn)
        if turn == 2:
            days = get_days(x=x, turn=turn)
            tau = 5.05 if self.patient else 3.03
            idx = f(0) if days <= tau else f(1)

        elif turn == 4:
            if self.patient:
                days = get_days(x=x, turn=turn)
                idx = f(0) if days <= 2.01 else f(.5)
            else:
                num_offers = get_recent_byr_offers(x=x, turn=turn)
                idx = f(1) if num_offers <= .5 else f(0)

        elif turn == 6:
            if self.patient:
                days4 = get_days(x=x, turn=4)
                if days4 <= 2.01:
                    days6 = get_days(x=x, turn=6)
                    idx = f(0) if days6 <= 2.04 else f(1)
                else:
                    norm = get_last_norm(x=x, turn=turn)
                    idx = f(.5) if norm <= .67 else f(1)
            else:
                idx = f(0)
        else:
            raise ValueError('Invalid turn: {}'.format(turn))

        # deterministic categorical action distribution
        pdf = torch.zeros(NUM_COMMON_CONS + 3, dtype=torch.float)
        pdf[idx] = 1.
        return pdf

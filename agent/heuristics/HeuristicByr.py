import torch
from agent.heuristics.util import get_agent_turn, wrapper
from agent.const import BYR_CONS, NUM_COMMON_CONS


class HeuristicByr:
    def __init__(self, index=None):
        self.cons = BYR_CONS.loc[index]

    def __call__(self, observation=None):
        # noinspection PyProtectedMember
        x = observation._asdict()

        # turn number
        turn = get_agent_turn(x=x, byr=True)
        assert turn in [1, 3, 5]

        # index of action
        f = wrapper(turn)
        idx = f(self.cons.loc[turn])

        # deterministic categorical action distribution
        pdf = torch.zeros(NUM_COMMON_CONS + 2, dtype=torch.float)
        pdf[idx] = 1.
        return pdf

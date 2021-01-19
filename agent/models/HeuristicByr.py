import torch
from agent.models.util import wrapper, get_agent_turn
from constants import NUM_COMMON_CONS


class HeuristicByr:

    def __call__(self, observation=None):
        # noinspection PyProtectedMember
        x = observation._asdict()

        # turn number
        turn = get_agent_turn(x=x, byr=True)

        # index of action
        f = wrapper(turn)
        if turn == 1:
            idx = f(.5)

        elif turn in [3, 5]:
            idx = f(.4)

        elif turn == 7:
            idx = f(1)
        else:
            raise ValueError('Invalid turn: {}'.format(turn))

        # deterministic categorical action distribution
        pdf = torch.zeros(NUM_COMMON_CONS + 2, dtype=torch.float)
        pdf[idx] = 1.
        return pdf

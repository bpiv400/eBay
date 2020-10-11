import torch
from constants import NUM_AGENT_CONS
from featnames import LSTG


class HeuristicByr:

    def __call__(self, observation=None):
        # noinspection PyProtectedMember
        input_dict = observation._asdict()

        # turn number
        turn_feats = input_dict[LSTG][-3:]
        turn = int(7 - 6 * turn_feats[0] - 4 * turn_feats[1] - 2 * turn_feats[2])

        # initialize output distribution with zeros
        pdf = torch.zeros(NUM_AGENT_CONS + 2, dtype=torch.float)

        if turn in [1, 3, 5]:
            pdf[1] = 1.  # minimal concession
        elif turn == 7:
            pdf[0] = 1.  # rejection
        else:
            raise ValueError('Invalid turn: {}'.format(turn))

        return pdf

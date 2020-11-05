import torch
from constants import NUM_COMMON_CONS, IDX
from featnames import LSTG, BYR, DELTA


class HeuristicByr:

    def __init__(self, **kwargs):
        self.delta = kwargs[DELTA]
        assert self.delta in [.9, .99]

    def __call__(self, observation=None):
        # noinspection PyProtectedMember
        input_dict = observation._asdict()

        # turn number
        turn_feats = input_dict[LSTG][-3:]
        turn = int(7 - 6 * turn_feats[0] - 4 * turn_feats[1] - 2 * turn_feats[2])
        assert turn in IDX[BYR]

        # initialize output distribution with zeros
        pdf = torch.zeros(NUM_COMMON_CONS + 2, dtype=torch.float)

        if turn == 1:
            pdf[1] = 1.  # 50%
        elif turn in [3, 5]:
            if self.delta == .9:
                pdf[1] = 1.  # 17%
            else:
                pdf[-3] = 1.  # 40%
        else:
            if self.delta == .9:
                pdf[0] = 1  # reject
            else:
                pdf[-1] = 1  # accept

        assert torch.isclose(torch.sum(pdf), torch.tensor(1.))

        return pdf

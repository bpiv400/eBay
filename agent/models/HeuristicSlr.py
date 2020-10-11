import torch
from rlenv.const import DAYS_IND
from constants import NUM_AGENT_CONS
from featnames import LSTG


class HeuristicSlr:

    def __call__(self, observation=None):
        # noinspection PyProtectedMember
        input_dict = observation._asdict()

        # turn number
        turn_feats = input_dict[LSTG][-2:]
        turn = int(6 - 4 * turn_feats[0] - 2 * turn_feats[1])

        # # extract other features
        # thread_num = input_dict[LSTG][-3] + 1
        # months = input_dict[LSTG][-5]
        # for i in range(1, turn + 1):
        #     months += input_dict['offer{}'.format(i)][DAYS_IND] / 31
        # assert months < 1.

        # initialize output distribution with zeros
        pdf = torch.zeros(NUM_AGENT_CONS + 3, dtype=torch.float)

        if turn == 2:
            pdf[0] = 1.
        elif turn in [4, 6]:
            pdf[-1] = 1.
        else:
            raise ValueError('Invalid turn: {}'.format(turn))

        # if turn == 2:
        #     if months > .55 and thread_num <= 2:
        #         pdf[-1] = 1.
        #     else:
        #         pdf[0] = 1.
        # elif turn == 4:
        #     if months > .51 and thread_num <= 3:
        #         pdf[-1] = 1.
        #     else:
        #         pdf[0] = 1.
        # elif turn == 6:
        #     if months > .31 and thread_num <= 6:
        #         pdf[-1] = 1.
        #     else:
        #         pdf[0] = 1.
        # else:
        #     raise ValueError('Invalid turn: {}'.format(turn))

        return pdf

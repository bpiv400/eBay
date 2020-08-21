import torch
from rlenv.const import DAYS_IND


class HeuristicSlr:

    def __init__(self):
        self.byr = False

    def __call__(self, observation=None):
        # initialize output distribution with zeros
        probs = torch.zeros(101, dtype=torch.float)

        # noinspection PyProtectedMember
        input_dict = observation._asdict()

        # turn number
        turn_feats = input_dict['lstg'][-2:]
        if turn_feats[0] == 1.:
            turn = 2
        elif turn_feats[1] == 1.:
            turn = 4
        else:
            turn = 6

        # extract other features
        thread_num = input_dict['lstg'][-3] + 1
        months = input_dict['lstg'][-5]
        for i in range(1, turn + 1):
            months += input_dict['offer{}'.format(i)][DAYS_IND] / 31
        assert months < 1.

        if turn == 2:
            if months > .55 and thread_num <= 2:
                probs[-1] = 1.
            else:
                probs[0] = 1.
        elif turn == 4:
            if months > .51 and thread_num <= 3:
                probs[-1] = 1.
            else:
                probs[0] = 1.
        elif turn == 6:
            if months > .31 and thread_num <= 6:
                probs[-1] = 1.
            else:
                probs[0] = 1.
        else:
            raise ValueError('Invalid turn: {}'.format(turn))

        return probs, None  # None for value

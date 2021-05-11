import numpy as np
import torch
from agent.const import DELTA_SLR, AGENT_CONS
from rlenv.const import TIME_FEATS, DAYS_IND, NORM_IND, BYR_OFFERS_RECENT_IND
from constants import IDX, NUM_COMMON_CONS, MAX_DAYS
from featnames import LSTG, BYR, SLR


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
                print(num_offers)
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


class HeuristicByr:
    def __init__(self, delta=None):
        assert delta in [.8, .9, 1, 1.5, 2]
        self.delta = delta

    def __call__(self, observation=None):
        # noinspection PyProtectedMember
        x = observation._asdict()

        # turn number
        turn = get_agent_turn(x=x, byr=True)

        # index of action
        f = wrapper(turn)
        if turn == 1:
            idx = f(.5) if self.delta < 2 else f(.6)

        elif turn == 3:
            if self.delta < .9:
                idx = f(.17)
            elif self.delta <= 1:
                idx = f(.4)
            else:
                idx = f(.5)

        elif turn == 5:
            if self.delta < 1:
                idx = f(.4)
            elif self.delta < 2:
                idx = f(.5)
            else:
                idx = f(1)

        elif turn == 7:
            if self.delta >= 1:
                idx = f(1)
            else:
                norm = get_last_norm(turn=turn, x=x)
                idx = f(0) if norm > self.delta else f(1)
        else:
            raise ValueError('Invalid turn: {}'.format(turn))

        # deterministic categorical action distribution
        pdf = torch.zeros(NUM_COMMON_CONS + 2, dtype=torch.float)
        pdf[idx] = 1.
        return pdf


def get_last_norm(turn=None, x=None):
    """
    Normalized last offer.
    :param dict x: input features
    :param int turn: turn number
    :return: float
    """
    norm_ind = NORM_IND
    if turn in IDX[BYR]:
        norm_ind -= len(TIME_FEATS)
    norm = x['offer{}'.format(turn - 1)][norm_ind].item()
    if turn in IDX[BYR]:
        norm = 1 - norm
    return norm


def get_days(x=None, turn=None):
    """
    Days since listing began.
    :param dict x: input features
    :param int turn: turn number
    :return: float
    """
    DAYS_SINCE_START_IND = -5
    days = x[LSTG][DAYS_SINCE_START_IND].item()
    if turn > 1:
        days_ind = DAYS_IND
        if turn in IDX[BYR]:
            days_ind -= len(TIME_FEATS)
        for t in range(2, turn + 1):
            days += x['offer{}'.format(t)][days_ind].item()
    assert days < MAX_DAYS
    return days


def get_recent_byr_offers(x=None, turn=None):
    """
    Number of buyer offers in last 48 hours.
    :param dict x: input features
    :param int turn: turn number
    :return: float
    """
    assert turn in IDX[SLR]
    num_offers = x['offer1'][BYR_OFFERS_RECENT_IND].item()
    for t in range(2, turn + 1):
        num_offers += x['offer{}'.format(t)][BYR_OFFERS_RECENT_IND].item()
    assert num_offers >= 0
    return num_offers


def wrapper(turn=None):
    return lambda x: np.nonzero(np.isclose(AGENT_CONS[turn], x))[0][0]


def get_turn(x, byr=None):
    last = 4 * x[:, -2] + 2 * x[:, -1]
    if byr:
        turn = 7 - 6 * x[:, -3] - last
    else:
        turn = 6 - last
    return int(turn.item())


def get_agent_turn(x=None, byr=None):
    turn_feats = x[LSTG][-3:] if byr else x[LSTG][-2:]
    turn_feats = turn_feats.unsqueeze(dim=0)
    return get_turn(x=turn_feats, byr=byr)

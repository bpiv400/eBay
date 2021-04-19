import numpy as np
import torch
from agent.const import DELTA_SLR, AGENT_CONS
from rlenv.const import TIME_FEATS, DAYS_IND, NORM_IND
from constants import MAX_DAYS, IDX, NUM_COMMON_CONS
from featnames import LSTG, BYR

DAYS_SINCE_START_IND = -5


class HeuristicSlr:
    def __init__(self, delta=None):
        self.high = np.isclose(delta, DELTA_SLR[-1])

    def __call__(self, observation=None):
        # noinspection PyProtectedMember
        x = observation._asdict()

        # turn number
        turn = get_agent_turn(x=x, byr=False)

        # index of action
        f = wrapper(turn)
        if turn == 2:
            elapsed = get_elapsed(x=x, turn=turn)
            tau = .49 if self.high else .38
            idx = f(0) if elapsed <= tau else f(1)

        elif turn == 4:
            if self.high:
                elapsed = get_elapsed(x=x, turn=turn)
                idx = f(0) if elapsed <= .33 else f(1)
            else:
                norm = get_last_norm(turn=turn, x=x)
                idx = f(.5) if norm <= .69 else f(1)

        elif turn == 6:
            if self.high:
                idx = f(1)
            else:
                norm = get_last_norm(turn=turn, x=x)
                idx = f(.5) if norm <= .65 else f(1)
        else:
            raise ValueError('Invalid turn: {}'.format(turn))

        # deterministic categorical action distribution
        pdf = torch.zeros(NUM_COMMON_CONS + 3, dtype=torch.float)
        pdf[idx] = 1.
        return pdf


class HeuristicByr:
    def __init__(self, delta=None):
        assert delta in [.75, 1., 1.5]
        self.delta = delta

    def __call__(self, observation=None):
        # noinspection PyProtectedMember
        x = observation._asdict()

        # turn number
        turn = get_agent_turn(x=x, byr=True)

        # index of action
        f = wrapper(turn)
        if turn == 1:
            if self.delta == 1.5:
                idx = f(.6)
            else:
                idx = f(.5)

        elif turn in [3, 5]:
            if self.delta == 1:
                idx = f(.4)
            elif self.delta == 1.5:
                idx = f(.5)
            else:
                idx = f(.17) if turn == 3 else f(.33)

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


def wrapper(turn=None):
    return lambda x: np.nonzero(np.isclose(AGENT_CONS[turn], x))[0][0]


def get_agent_turn(x=None, byr=None):
    turn_feats = x[LSTG][-3:] if byr else x[LSTG][-2:]
    turn_feats = turn_feats.unsqueeze(dim=0)
    turn = int(get_turn(x=turn_feats, byr=byr).item())
    return turn


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


def get_elapsed(x=None, turn=None):
    """
    Fraction of listing window elapsed.
    :param dict x: input features
    :param int turn: turn number
    :return: float
    """
    elapsed = x[LSTG][DAYS_SINCE_START_IND].item() / MAX_DAYS
    if turn > 1:
        days_ind = DAYS_IND
        if turn in IDX[BYR]:
            days_ind -= len(TIME_FEATS)
        for i in range(2, turn + 1):
            elapsed += x['offer{}'.format(i)][days_ind].item() / MAX_DAYS
    assert elapsed < 1.
    return elapsed


def get_turn(x, byr=None):
    last = 4 * x[:, -2] + 2 * x[:, -1]
    if byr:
        return 7 - 6 * x[:, -3] - last
    else:
        return 6 - last

import numpy as np
from agent.const import AGENT_CONS
from env.const import NORM_IND, DAYS_IND, BYR_OFFERS_RECENT_IND
from constants import IDX, MAX_DAYS
from featnames import SLR, LSTG, BYR, TIME_FEATS

DAYS_SINCE_START_IND = -5


def get_last_norm(turn=None, x=None):
    """
    Normalized last offer.
    :param dict x: input features
    :param int turn: turn number
    :return: float
    """
    assert turn in IDX[SLR]
    norm = 1 - x['offer{}'.format(turn - 1)][NORM_IND].item()
    return norm


def get_days(x=None, turn=None):
    """
    Days since listing began.
    :param dict x: input features
    :param int turn: turn number
    :return: float
    """
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
    last = 5 if byr else 6
    turn = last - (4 * x[:, 0] + 2 * x[:, 1])
    return int(turn.item())


def get_agent_turn(x=None, byr=None):
    turn_feats = x[LSTG][-2:].unsqueeze(dim=0)
    return get_turn(x=turn_feats, byr=byr)

import numpy as np
from agent.util import get_turn
from rlenv.const import TIME_FEATS, DAYS_IND, NORM_IND, AUTO_IND, EXP_IND
from agent.models.const import DAYS_SINCE_START_IND, STORE_IND, \
    START_PRICE_PCTILE_IND, START_PRICE_PCTILES
from agent.const import AGENT_CONS
from constants import MAX_DAYS, IDX
from featnames import LSTG, BYR


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


def get_last_auto(x=None, turn=None):
    """
    Indicator for whether last offer was an auto-reject.
    :param dict x: input features
    :param int turn: turn number
    :return: float
    """
    assert turn in [3, 5, 7]
    auto_ind = AUTO_IND - len(TIME_FEATS)
    auto = x['offer{}'.format(turn - 1)][auto_ind].item()
    return auto


def get_last_exp(x=None, turn=None):
    """
    Indicator for whether last offer was an expiration reject.
    :param dict x: input features
    :param int turn: turn number
    :return: float
    """
    assert turn in [3, 5, 7]
    exp_ind = EXP_IND - len(TIME_FEATS)
    exp = x['offer{}'.format(turn - 1)][exp_ind].item()
    return exp


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


def get_store(x=None):
    return bool(x[LSTG][STORE_IND].item())


def get_start_price(x=None):
    p = x[LSTG][START_PRICE_PCTILE_IND].item()
    price = START_PRICE_PCTILES[np.isclose(START_PRICE_PCTILES, p)].index[0]
    return price
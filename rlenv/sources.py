import math
import numpy as np, pandas as pd
from datetime import datetime as dt
from rlenv.env_consts import *
from rlenv.env_utils import featname, last_norm
from featnames import *


class Sources:
    def __init__(self, x_lstg=None):
        self.source_dict = x_lstg
        self.source_dict[MONTHS_SINCE_LSTG] = 0.0

    def __call__(self):
        return self.source_dict


class ThreadSources(Sources):
    def __init__(self, x_lstg=None):
        super(ThreadSources, self).__init__(x_lstg=x_lstg)
        # other clock features initialized to lstg start date
        self.source_dict[TURN_IND_MAP] = ThreadSources._turn_inds(1)
        self.source_dict[INT_REMAINING] = 0.0
        self.source_dict[BYR_HIST] = 0.0
        self.offer_prev_time = None

    def prepare_hist(self, time_feats=None, clock_feats=None, months_since_lstg=None):
        self.source_dict[MONTHS_SINCE_LSTG] = months_since_lstg
        self.source_dict[INIT_CLOCK] = clock_feats
        self.source_dict[INIT_TIME] = time_feats
        self.offer_prev_time = time_feats

    def init_thread(self, hist=None):
        # (time features and clock_feats already set during prepare_hist)
        # add byr history
        self.source_dict[BYR_HIST] = hist
        del self.source_dict[INIT_CLOCK]
        del self.source_dict[INIT_TIME]
        


    def update_offer(self, outcomes=None, turn=None):
        outcome_names = ALL_OUTCOMES[turn].copy()
        if turn > 1:
            outcomes[featname(DAYS, turn)] = self.source_dict[THREAD_MAP][featname(DAYS, turn)]
            outcomes[featname(DELAY, turn)] = self.source_dict[THREAD_MAP][featname(DELAY, turn)]
        # remove deterministic outcomes
        if turn == 1:
            outcome_names.remove(featname(DAYS, turn))
            outcome_names.remove(featname(DELAY, turn))
        elif turn == 7:
            outcome_names.remove(featname(MSG, turn))

        self.source_dict[THREAD_MAP][outcome_names] = outcomes[outcome_names]
        return outcomes[featname(NORM, turn)]

    def change_turn(self, turn):
        # turn indicator
        self.source_dict[TURN_IND_MAP] = ThreadSources._turn_inds(turn)

    def init_offer(self, time_feats=None, clock_feats=None, turn=None):
        # NOTE : Not called on turn 1
        time_diff = time_feats - self.offer_prev_time
        self.offer_prev_time = time_feats
        self.source_dict[THREAD_MAP][ALL_CLOCK_FEATS[turn]] = clock_feats
        self.source_dict[THREAD_MAP][ALL_TIME_FEATS[turn]] = time_diff

    def prepare_offer(self, days=None, delay=None, turn=None):
        self.source_dict[THREAD_MAP][featname(DAYS, turn)] = days
        self.source_dict[THREAD_MAP][featname(DELAY, turn)] = delay
        self.delay_prev_time = None

    def byr_expire(self, turn=None):
        norm = last_norm(self(), turn=turn)
        self.source_dict[THREAD_MAP][featname(NORM, turn)] = norm

    def init_remaining(self, remaining):
        self.source_dict[THREAD_MAP][INT_REMAINING] = remaining

    def is_sale(self, turn):
        return self.source_dict[THREAD_MAP][featname(CON, turn)] == 1

    def is_rej(self, turn):
        return self.source_dict[THREAD_MAP][featname(CON, turn)] == 0

    def is_expire(self, turn):
        return self.source_dict[THREAD_MAP][featname(DELAY, turn)] == 1

    def summary(self, turn):
        con = int(self.source_dict[THREAD_MAP][featname(CON, turn)] * 100)
        norm = self.source_dict[THREAD_MAP][featname(NORM, turn)] * 100
        norm = norm.round()
        if turn != 7:
            msg = self.source_dict[THREAD_MAP][featname(MSG, turn)] == 1
        else:
            msg = False
        split = self.source_dict[THREAD_MAP][featname(SPLIT, turn)] == 1
        return con, norm, msg, split

    def get_delay_outcomes(self, turn):
        days = self.source_dict[THREAD_MAP][featname(DAYS, turn)]
        delay = self.source_dict[THREAD_MAP][featname(DELAY, turn)]
        return days, delay

    def get_slr_outcomes(self, turn):
        auto = self.source_dict[THREAD_MAP][featname(AUTO, turn)]
        rej = self.source_dict[THREAD_MAP][featname(REJECT, turn)]
        exp = self.source_dict[THREAD_MAP][featname(EXP, turn)]
        return auto, exp, rej

    @staticmethod
    def _turn_inds(turn):
        if turn % 2 == 0:
            vec = np.zeros((2, ), dtype=np.float)
        else:
            vec = np.zeros((3, ), dtype=np.float)
        if turn <= 5:
            ind = math.floor((turn - 1) / 2)
            vec[ind] = 1
        return vec


class ArrivalSources(Sources):
    def __init__(self, x_lstg=None, composer=None):
        super(ArrivalSources, self).__init__(x_lstg=x_lstg, composer=composer)

    def update_arrival(self, clock_feats=None, months_since_lstg=None, thread_count=None):
        prev_months_since_lstg = self.source_dict[THREAD_MAP][MONTHS_SINCE_LSTG]
        months_since_last = months_since_lstg - prev_months_since_lstg
        self.source_dict[THREAD_MAP][THREAD_COUNT] = thread_count
        self.source_dict[THREAD_MAP][CLOCK_FEATS] = clock_feats
        self.source_dict[THREAD_MAP][MONTHS_SINCE_LAST] = months_since_last

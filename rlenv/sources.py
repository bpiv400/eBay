import math
import numpy as np
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
        for i in range(1, 8):
            self.source_dict[OFFER_MAPS[i]] = np.zeros(len(ALL_OFFER_FEATS),
                                                       dtype=np.float)
        self.offer_prev_time = None

    def prepare_hist(self, time_feats=None, clock_feats=None, months_since_lstg=None):
        self.source_dict[MONTHS_SINCE_LSTG] = months_since_lstg
        offer_map = OFFER_MAPS[1]
        self.source_dict[offer_map][CLOCK_START_IND:CLOCK_END_IND] = clock_feats
        self.source_dict[offer_map][TIME_START_IND:TIME_END_IND] = time_feats
        self.offer_prev_time = time_feats

    def init_thread(self, hist=None):
        # (time features and clock_feats already set during prepare_hist)
        # add byr history
        self.source_dict[BYR_HIST] = hist

    def update_offer(self, offer_outcomes=None, turn=None):
        offer_map = OFFER_MAPS[turn]
        self.source_dict[offer_map][CON_START_IND] = offer_outcomes
        return self.source_dict[offer_map][NORM_IND]

    def init_offer(self, time_feats=None, clock_feats=None, turn=None):
        # NOTE : Not called on turn 1
        time_diff = time_feats - self.offer_prev_time
        offer_map = OFFER_MAPS[turn]
        self.offer_prev_time = time_feats
        feats = np.concatenate([clock_feats, time_diff])
        self.source_dict[offer_map][CLOCK_START_IND:TIME_END_IND] = feats

    def prepare_offer(self, delay_outcomes=None, turn=None):
        offer_map = OFFER_MAPS[turn]
        self.source_dict[offer_map][DELAY_START_IND:DELAY_END_IND] = delay_outcomes

    def byr_expire(self, turn=None):
        norm = last_norm(self(), turn=turn)
        offer_map = OFFER_MAPS[turn]
        self.source_dict[offer_map][NORM_IND] = norm

    def init_remaining(self, remaining):
        self.source_dict[INT_REMAINING] = remaining

    def is_sale(self, turn):
        offer_map = OFFER_MAPS[turn]
        return self.source_dict[offer_map][CON_IND] == 1

    def is_rej(self, turn):
        offer_map = OFFER_MAPS[turn]
        return self.source_dict[offer_map][CON_IND] == 0

    def is_expire(self, turn):
        offer_map = OFFER_MAPS[turn]
        return self.source_dict[offer_map][DELAY_IND] == 1

    def summary(self, turn):
        # TODO: could index once and pass resulting vector to output
        # would require update of recorders
        offer_map = OFFER_MAPS[turn]
        con = int(self.source_dict[offer_map][featname(CON, turn)] * 100)
        norm = self.source_dict[offer_map][featname(NORM, turn)] * 100
        norm = norm.round()
        msg = self.source_dict[offer_map][featname(MSG, turn)] == 1

        split = self.source_dict[offer_map][featname(SPLIT, turn)] == 1
        return con, norm, msg, split

    def get_delay_outcomes(self, turn):
        # see update in sources.summary for slight efficiency improvement
        offer_map = OFFER_MAPS[turn]
        days = self.source_dict[offer_map][DAYS_IND]
        delay = self.source_dict[offer_map][DELAY_IND]
        return days, delay

    def get_slr_outcomes(self, turn):
        # see update in sources.summary for slight efficiency improvement
        offer_map = OFFER_MAPS[turn]
        auto = self.source_dict[offer_map][AUTO_IND]
        rej = self.source_dict[offer_map][REJECT_IND]
        exp = self.source_dict[offer_map][EXP_IND]
        return auto, exp, rej
    

class ArrivalSources(Sources):
    def __init__(self, x_lstg=None, composer=None):
        super(ArrivalSources, self).__init__(x_lstg=x_lstg, composer=composer)

    def update_arrival(self, clock_feats=None, months_since_lstg=None, thread_count=None):
        prev_months_since_lstg = self.source_dict[THREAD_MAP][MONTHS_SINCE_LSTG]
        months_since_last = months_since_lstg - prev_months_since_lstg
        self.source_dict[THREAD_MAP][THREAD_COUNT] = thread_count
        self.source_dict[THREAD_MAP][CLOCK_FEATS] = clock_feats
        self.source_dict[THREAD_MAP][MONTHS_SINCE_LAST] = months_since_last

import math
import numpy as np
import pandas as pd
from rlenv.env_consts import (TIME_FEATS, ALL_TIME_FEATS,
                              ALL_CLOCK_FEATS, BYR_HIST, DAYS, DELAY, ALL_OUTCOMES,
                              NORM, TURN_FEATS, MONTHS_SINCE_LSTG, CLOCK_FEATS,
                              DURATION, INT_REMAINING, CON, MSG, SPLIT, EXP, REJECT, AUTO)
from rlenv.composer.maps import *
from rlenv.env_utils import featname


class Sources:
    def __init__(self, x_lstg=None, composer=None):
        self.source_dict = {
            LSTG_MAP: x_lstg,
            X_TIME_MAP: pd.Series(0.0, index=composer.feat_sets[X_TIME_MAP])
        }

    def __call__(self):
        return self.source_dict


# TODO SPLIT SOURCES INTO TWO CLASSES
class ThreadSources(Sources):
    def __init__(self, x_lstg=None, composer=None):
        super(ThreadSources, self).__init__(x_lstg=x_lstg, composer=composer)
        # other clock features initialized to lstg start date
        self.source_dict[TURN_IND_MAP] = pd.Series(0.0, index=composer.feat_sets[TURN_IND_MAP])
        self.source_dict[THREAD_MAP] = pd.Series(0.0, index=composer.feat_sets[THREAD_MAP])
        self.delay_prev_time = None
        self.offer_prev_time = None

    def prepare_hist(self, time_feats=None, clock_feats=None, months_since_lstg=None):
        self.source_dict[THREAD_MAP][MONTHS_SINCE_LSTG] = months_since_lstg
        self.source_dict[THREAD_MAP][ALL_CLOCK_FEATS[1]] = clock_feats
        self.source_dict[THREAD_MAP][ALL_TIME_FEATS[1]] = time_feats
        self.offer_prev_time = time_feats

    def init_thread(self, hist=None):
        # (time features and clock_feats already set during prepare_hist)
        # add byr history
        self.source_dict[THREAD_MAP][BYR_HIST] = hist
        # initial turn indices to buyer indices and activate turn 1
        self.source_dict[TURN_IND_MAP]['t1'] = 1

    def update_offer(self, outcomes=None, turn=None):
        print('outcomes')
        print(outcomes[featname(DAYS, turn)])
        print('source')
        for feat in self.source_dict[THREAD_MAP].index:
            if '_1' in feat:
                print(feat)
        print(self.source_dict[THREAD_MAP].index)
        print(self.source_dict[THREAD_MAP][featname(DAYS, turn)])

        outcomes[featname(DAYS, turn)] = self.source_dict[THREAD_MAP][featname(DAYS, turn)]
        outcomes[featname(DELAY, turn)] = self.source_dict[THREAD_MAP][featname(DELAY, turn)]
        self.source_dict[ALL_OUTCOMES[turn]] = outcomes
        return outcomes[featname(NORM, turn)]

    def change_turn(self, turn):
        # turn indicator
        self.source_dict[TURN_IND_MAP] = ThreadSources._turn_inds(turn)

    def init_delay(self, months_since_lstg=None):
        # TODO: Note, we've assumed months_since_lstg is in the fixed features rather than the time features
        self.source_dict[THREAD_MAP][MONTHS_SINCE_LSTG] = months_since_lstg

    def update_delay(self, time_feats=None, remaining=None, duration=None,
                     clock_feats=None):
        if self.delay_prev_time is None:
            time_diff = np.zeros(len(TIME_FEATS))
        else:
            time_diff = time_feats - self.delay_prev_time
        self.source_dict[X_TIME_MAP][TIME_FEATS] = time_diff
        self.delay_prev_time = time_feats
        self.source_dict[X_TIME_MAP][INT_REMAINING] = remaining
        self.source_dict[X_TIME_MAP][DURATION] = duration
        self.source_dict[X_TIME_MAP][CLOCK_FEATS] = clock_feats

    def init_offer(self, time_feats=None, clock_feats=None, turn=None):
        # NOTE : Not called on turn 1
        time_diff = time_feats - self.delay_prev_time
        self.offer_prev_time = time_feats
        self.source_dict[THREAD_MAP][ALL_CLOCK_FEATS[turn]] = clock_feats
        self.source_dict[THREAD_MAP][ALL_TIME_FEATS[turn]] = time_diff

    def prepare_offer(self, days=None, delay=None, turn=None):
        self.source_dict[THREAD_MAP][featname(DAYS, turn)] = days
        self.source_dict[THREAD_MAP][featname(DELAY, turn)] = delay
        self.delay_prev_time = None

    def byr_expire(self, days=None, turn=None):
        if turn == 1:
            prev_norm = 0.0
        else:
            prev_norm = self.source_dict[THREAD_MAP][featname(NORM, turn - 2)]
        # updating outcomes
        self.source_dict[THREAD_MAP][featname(NORM, turn)] = prev_norm
        self.source_dict[THREAD_MAP][featname(DAYS, turn)] = days
        self.source_dict[THREAD_MAP][featname(DELAY, turn)] = 1

    def is_sale(self, turn):
        return self.source_dict[THREAD_MAP][featname(CON, turn)] == 1

    def is_rej(self, turn):
        return self.source_dict[THREAD_MAP][featname(CON, turn)] == 0

    def summary(self, turn):
        con = int(self.source_dict[THREAD_MAP][featname(CON, turn)] * 100)
        norm = self.source_dict[THREAD_MAP][featname(NORM, turn)] * 100
        norm = norm.round()
        msg = self.source_dict[THREAD_MAP][MSG] == 1
        split = self.source_dict[THREAD_MAP][SPLIT]
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
        vec = pd.Series(0.0, index=TURN_FEATS)
        if turn <= 5:
            ind = math.ceil((turn / 2))
            vec['t{}'.format(ind)] = 1
        return vec


class ArrivalSources(Sources):
    def __init__(self, x_lstg=None, composer=None):
        super(ArrivalSources, self).__init__(x_lstg=x_lstg, composer=composer)
        self.prev_time = np.zeros(len(TIME_FEATS))

    def update_arrival(self, time_feats=None, clock_feats=None, duration=None):
        self.source_dict[X_TIME_MAP][DURATION] = duration
        self.source_dict[X_TIME_MAP][CLOCK_FEATS] = clock_feats
        self.source_dict[X_TIME_MAP][TIME_FEATS] = time_feats - self.prev_time
        self.prev_time = time_feats

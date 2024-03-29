from copy import deepcopy
import numpy as np
from env.util import last_norm
from env.const import OFFER_MAPS, CLOCK_START_IND, TIME_END_IND, NORM_IND, \
    MSG_IND, DELAY_START_IND, DELAY_END_IND, CON_IND, DELAY_IND, CON_START_IND, \
    DAYS_IND, AUTO_IND, REJECT_IND, EXP_IND
from featnames import DAYS_SINCE_LSTG, INT_REMAINING, BYR_HIST, ALL_OFFER_FEATS


class Sources:
    def __init__(self, x_lstg=None):
        self.source_dict = deepcopy(x_lstg)
        self.source_dict[DAYS_SINCE_LSTG] = 0.0

    def __call__(self):
        return self.source_dict


class ThreadSources(Sources):
    def __init__(self, x_lstg=None, hist_pctile=None, days_since_lstg=None):
        super().__init__(x_lstg=x_lstg)
        # other clock features initialized to lstg start date
        self.source_dict[INT_REMAINING] = 0.0
        self.source_dict[BYR_HIST] = hist_pctile
        self.source_dict[DAYS_SINCE_LSTG] = days_since_lstg
        for i in range(1, 8):
            self.source_dict[OFFER_MAPS[i]] = np.zeros(len(ALL_OFFER_FEATS),
                                                       dtype=np.float)
        self.offer_prev_time = None

    def update_con_outcomes(self, con_outcomes=None, turn=None):
        offer_map = OFFER_MAPS[turn]
        self.source_dict[offer_map][CON_START_IND:MSG_IND] = con_outcomes
        # print('summary after update')
        # print(self.summary(turn=turn))
        return self.source_dict[offer_map][NORM_IND]

    def update_msg(self, msg=None, turn=None):
        offer_map = OFFER_MAPS[turn]
        self.source_dict[offer_map][MSG_IND] = msg

    def init_offer(self, time_feats=None, clock_feats=None, turn=None):
        offer_map = OFFER_MAPS[turn]
        if turn == 1:
            time_diff = time_feats
        else:
            time_diff = time_feats - self.offer_prev_time
        self.offer_prev_time = time_feats
        feats = np.concatenate([clock_feats, time_diff])
        self.source_dict[offer_map][CLOCK_START_IND:TIME_END_IND] = feats

    def update_delay(self, delay_outcomes=None, turn=None):
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
        # TODO: would require update of recorders
        offer_map = OFFER_MAPS[turn]
        con = int(np.round(self.source_dict[offer_map][CON_IND] * 100))
        msg = self.source_dict[offer_map][MSG_IND] == 1
        norm = self.source_dict[offer_map][NORM_IND]
        return con, norm, msg

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

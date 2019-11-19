import math
import torch
from rlenv.env_consts import (TIME_FEATS, BYR_OUTCOMES, SLR_OUTCOMES, NORM_POS,
                              DAYS_POS, DELAY_POS, CON_POS, MSG_POS)
from rlenv.composer.maps import *
from rlenv.env_utils import get_clock_feats


class Sources:
    def __init__(self):
        self.source_dict = None

    def __call__(self):
        return self.source_dict


# TODO SPLIT SOURCES INTO TWO CLASSES
class ThreadSources(Sources):
    def __init__(self, x_lstg=None, start_date=None):
        super(ThreadSources, self).__init__()
        # other clock features initialized to lstg start date
        self.source_dict[O_CLOCK_MAP] = get_clock_feats(start_date)
        self.source_dict[LSTG_MAP] = x_lstg
        self.delay_prev_time = None

    def prepare_hist(self, time_feats=None, clock_feats=None, months_since_lstg=None):
        self.source_dict[MONTHS_LSTG_MAP] = months_since_lstg
        self.source_dict[CLOCK_MAP] = clock_feats
        self.source_dict[TIME_MAP] = time_feats

    def init_thread(self, hist=None, time_feats=None):
        # time features updated to exclude focal thread
        self.source_dict[TIME_MAP] = time_feats
        # add byr history
        self.source_dict[BYR_HIST_MAP] = hist
        # other clock features initialized to lstg start date
        self.source_dict[O_CLOCK_MAP] = get_clock_feats(self.start_date)
        # differences initialized to raw time features because all features were 0 at lstg start
        self.source_dict[DIFFS_MAP] = self.source_dict[TIME_MAP].clone()
        # differences between turn 0 and -1 are 0
        self.source_dict[O_DIFFS_MAP] = torch.zeros(len(TIME_FEATS))
        # initial turn indices to buyer indices and activate turn 1
        self.source_dict[TURN_IND_MAP] = ThreadSources._turn_inds(1)
        # initialize last outcomes to all 0's
        self.source_dict[L_OUTCOMES_MAP] = ThreadSources._init_pre_lstg_outcomes()
        # initialize last outcome to have all 0's except auto and rej
        self.source_dict[O_OUTCOMES_MAP] = ThreadSources._init_lstg_outcomes()
        # day, days outcomes to 0
        self.source_dict[OUTCOMES_MAP] = ThreadSources._init_outcomes(1)

    def update_offer(self, outcomes=None):
        outcomes[[DAYS_POS, DELAY_POS]] = self.source_dict[OUTCOMES_MAP][[DAYS_POS, DELAY_POS]]
        self.source_dict[OUTCOMES_MAP] = outcomes
        return outcomes[NORM_POS]

    def change_turn(self, turn):
        # push other sources
        self.source_dict[L_TIME_MAP] = self.source_dict[O_TIME_MAP]
        self.source_dict[L_CLOCK_MAP] = self.source_dict[O_CLOCK_MAP]
        self.source_dict[L_OUTCOMES_MAP] = self.source_dict[O_OUTCOMES_MAP]
        # push current sources
        self.source_dict[O_TIME_MAP] = self.source_dict[TIME_MAP]
        self.source_dict[O_OUTCOMES_MAP] = self.source_dict[OUTCOMES_MAP]
        self.source_dict[O_CLOCK_MAP] = self.source_dict[CLOCK_MAP]
        self.source_dict[O_DIFFS_MAP] = self.source_dict[DIFFS_MAP]
        # turn indicator
        self.source_dict[TURN_IND_MAP] = ThreadSources._turn_inds(turn)
        # new outcomes
        self.source_dict[OUTCOMES_MAP] = ThreadSources._init_outcomes(turn)

    def init_delay(self, months_since_lstg=None, days_since_thread=None,
                   time_feats=None):
        self.source_dict[MONTHS_LSTG_MAP] = months_since_lstg
        self.source_dict[DAYS_THREAD_MAP] = days_since_thread
        self.delay_prev_time = time_feats

    def update_delay(self, time_feats=None, remaining=None, duration=None,
                     clock_feats=None):
        self.source_dict[DIFFS_MAP] = time_feats - self.delay_prev_time
        self.delay_prev_time = time_feats
        self.source_dict[INT_REMAIN_MAP] = remaining
        self.source_dict[DUR_MAP] = duration
        self.source_dict[CLOCK_MAP] = clock_feats

    def init_offer(self, time_feats=None, clock_feats=None):
        self.source_dict[TIME_MAP] = time_feats
        self.source_dict[CLOCK_MAP] = clock_feats
        self.source_dict[DIFFS_MAP] = time_feats - self.source_dict[O_TIME_MAP]

    def prepare_offer(self, days=None, delay=None):
        input_vec = torch.tensor([days, delay]).float()
        self.source_dict[OUTCOMES_MAP][[DAYS_POS, DELAY_POS]] = input_vec
        self.delay_prev_time = None

    def byr_expire(self):
        self.source_dict[OUTCOMES_MAP][NORM_POS] = self.source_dict[L_OUTCOMES_MAP][NORM_POS]
        self.source_dict[OUTCOMES_MAP][CON_POS, MSG_POS] = 0

    def is_sale(self):
        return self.source_dict[OUTCOMES_MAP][CON_POS] == 1

    def is_rej(self):
        return self.source_dict[OUTCOMES_MAP][CON_POS] == 0

    def summary(self):
        con = int(self.source_dict[OUTCOMES_MAP][CON_POS] * 100)
        norm = self.source_dict[OUTCOMES_MAP][NORM_POS] * 100
        norm = round(norm)
        msg = self.source_dict[OUTCOMES_MAP][MSG_POS] == 1
        return con, norm, msg

    @staticmethod
    def _init_pre_lstg_outcomes():
        outcomes = torch.zeros(len(BYR_OUTCOMES)).float()
        return outcomes

    @staticmethod
    def _init_lstg_outcomes():
        outcomes = torch.zeros(len(SLR_OUTCOMES)).float()
        return outcomes

    @staticmethod
    def _init_outcomes(turn):
        if turn % 2 == 0:
            outcomes = torch.zeros(len(SLR_OUTCOMES)).float()
        else:
            outcomes = torch.zeros(len(BYR_OUTCOMES)).float()
        return outcomes

    @staticmethod
    def _turn_inds(turn):
        if turn % 2 == 0:
            num_turns = 2
        else:
            num_turns = 3
        vec = torch.zeros(num_turns).float()
        if turn < 5:
            ind = math.floor(turn / 2)
            vec[ind] = 1
        return vec


class ArrivalSources(Sources):
    def __init__(self):
        super(ArrivalSources, self).__init__()
        self.source_dict[DIFFS_MAP] = torch.zeros(len(TIME_FEATS)).float()

    def update_arrival(self, time_feats=None, clock_feats=None, duration=None):
        self.source_dict[DUR_MAP] = duration
        self.source_dict[CLOCK_MAP] = clock_feats
        self.source_dict[DIFFS_MAP] = time_feats - self.source_dict[TIME_MAP]
        self.source_dict[TIME_MAP] = time_feats







import torch
from rlenv.env_consts import (TIME_FEATS, BYR_OUTCOMES, SLR_OUTCOMES, AUTO_POS,
                              REJ_POS, AUTO_REJ_LSTG)
from rlenv.composer.maps import *
from constants import ARRIVAL_PERIODS
from rlenv.env_utils import get_clock_feats

INCREMENTER = torch.tensor([1]).float()

class Sources:
    def __init__(self, num_offers=False, x_lstg=None, start_date=None):
        self.num_offers = num_offers
        self.period = torch.tensor([-1]).float()
        if num_offers:
            self.max_periods = torch.tensor([ARRIVAL_PERIODS]).float()
        else:
            self.max_periods = None
        self.start_date = start_date
        self.source_dict = {
            LSTG_MAP: x_lstg,
            DIFFS_MAP: torch.zeros(len(TIME_FEATS)).float()
        }

    def __call__(self):
        return self.source_dict

    def update_arrival(self, time_feats=None, clock_feats=None):
        self.period = self.period + INCREMENTER
        self.source_dict[DUR_MAP] = self.period / self.max_periods
        self.source_dict[CLOCK_MAP] = clock_feats
        self.source_dict[DIFFS_MAP] = time_feats - self.source_dict[DIFFS_MAP]

    def prepare_hist(self, time_feats=None, clock_feats=None, months_since_lstg=None):
        self.source_dict[MONTHS_LSTG_MAP] = months_since_lstg
        self.source_dict[CLOCK_MAP] = clock_feats
        self.source_dict[TIME_MAP] = time_feats

    def init_offer(self, hist=None, time_feats=None):
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
        self.source_dict[TURN_IND_MAP] = Sources._init_inds()
        # initialize last outcomes to all 0's
        self.source_dict[L_OUTCOMES_MAP] = Sources._init_pre_lstg_outcomes()
        # initialize last outcome to have all 0's except auto and rej
        self.source_dict[O_OUTCOMES_MAP] = Sources._init_lstg_outcomes()
        # day, days outcomes to 0
        self.source_dict[OUTCOMES_MAP] = Sources._init_first_outcomes()

    @staticmethod
    def _init_inds():
        inds = torch.zeros(3).float()
        inds[0] = 1
        return inds

    @staticmethod
    def _init_pre_lstg_outcomes():
        outcomes = torch.zeros(len(BYR_OUTCOMES)).float()
        return outcomes

    @staticmethod
    def _init_lstg_outcomes():
        outcomes = torch.zeros(len(SLR_OUTCOMES)).float()
        outcomes[[AUTO_POS, REJ_POS]] = AUTO_REJ_LSTG
        return outcomes

    @staticmethod
    def _init_first_outcomes():
        outcomes = torch.zeros(len(BYR_OUTCOMES)).float()
        return outcomes







import torch
from rlenv.env_consts import CLOCK_MAP, LSTG_MAP, DIFFS_MAP, DUR_MAP, TIME_FEATS
from constants import ARRIVAL_PERIODS
import utils

INCREMENTER = torch.tensor([1]).float()

class Sources:
    def __init__(self, arrival=False, start_date=0, x_lstg=None):
        self.arrival = arrival
        self.start_date = start_date
        self.period = torch.tensor([-1]).float()
        if arrival:
            self.max_periods = torch.tensor([ARRIVAL_PERIODS]).float()
        self.source_dict = {
            LSTG_MAP: x_lstg,
            DIFFS_MAP: torch.zeros(len(TIME_FEATS)).float()
        }

    def __call__(self):
        return self.source_dict

    def update_arrival(self, time_feats=None, time=None):
        self.period = self.period + INCREMENTER
        self.source_dict[DUR_MAP] = self.period / self.max_periods
        self.source_dict[CLOCK_MAP] = utils.get_clock_feats(time, self.start_date,
                                                            arrival=self.arrival,
                                                            delay=False)
        self.source_dict[DIFFS_MAP] = time_feats - TIME_FEATS


    def init_offer(self, time_feats=None, time=None):

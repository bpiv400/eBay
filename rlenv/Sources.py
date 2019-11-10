import torch
from rlenv.env_consts import CLOCK_MAP, LSTG_MAP, TIME_MAP, DUR_MAP
from constants import ARRIVAL_PERIODS
import utils


class Sources:
    def __init__(self, arrival=False, start_date=0, x_lstg=None):
        self.arrival = arrival
        self.start_date = start_date
        self.period = torch.tensor([-1]).float()
        if arrival:
            self.max_periods = torch.tensor([ARRIVAL_PERIODS]).float()
        self.source_dict = {
            LSTG_MAP: x_lstg
        }

    def __call__(self):
        return self.source_dict

    def update_arrival(self, time_feats=None, time=None):
        self.period = self.period + 1
        self.source_dict[DUR_MAP] = self.period / self.max_periods
        self.source_dict[CLOCK_MAP] = utils.get_clock_feats(time, self.start_date,
                                                            arrival=self.arrival,
                                                            delay=False)
        self.source_dict[TIME_MAP] = time_feats

    def init_offer(self, time_feats=None, time=None):

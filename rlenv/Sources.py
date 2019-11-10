from rlenv.env_consts import CLOCK_MAP, LSTG_MAP, TIME_MAP
import utils


class Sources:
    def __init__(self, arrival=False, start_date=0, x_lstg=None):
        self.arrival = arrival
        self.start_date = start_date
        self.source_dict = {
            LSTG_MAP: x_lstg
        }

    def __call__(self):
        return self.source_dict

    def update_arrival(self, time_feats=None, time=None):
        self.source_dict[CLOCK_MAP] = utils.get_clock_feats(time, self.start_date,
                                                            arrival=self.arrival,
                                                            delay=False)
        self.source_dict[TIME_MAP] = time_feats

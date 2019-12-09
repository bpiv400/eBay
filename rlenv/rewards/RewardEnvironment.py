import numpy as np
from rlenv.time import time_triggers
from rlenv.env_consts import (MONTH, START_DAY, ACC_PRICE, VERBOSE)
from rlenv.env_utils import get_clock_feats
from rlenv.simulators import SimulatedSeller, SimulatedBuyer
from rlenv.events.RewardThread import RewardThread
from rlenv.environments.EbayEnvironment import EbayEnvironment


class RewardEnvironment(EbayEnvironment):
    def __init__(self, **kwargs):
        super(RewardEnvironment, self).__init__(kwargs['arrival'])
        # environment interface
        self.buyer = kwargs['buyer']
        self.seller = kwargs['seller']

        # features
        self.x_lstg = kwargs['x_lstg']
        self.lookup = kwargs['lookup']

        # end time
        self.end_time = self.lookup[START_DAY] + MONTH
        self.thread_counter = 0

        # recorder
        self.recorder = kwargs['recorder']

    def reset(self):
        super(RewardEnvironment, self).reset()
        self.recorder.reset_sim()
        if VERBOSE:
            print('Initializing Simulation {}'.format(self.recorder.sim))

    def run(self):
        """
        Runs a simulation of a single lstg until sale or expiration

        :return: a 3-tuple of (bool, float, int) giving whether the listing sells,
        the amount it sells for if it sells, and the amount of time it took to sell
        """
        super(RewardEnvironment, self).run()
        return self.outcome

    def _record(self, event, start_thread=False, byr_hist=None):
        if not start_thread:
            self.recorder.add_offer(event)
        else:
            self.recorder.start_thread(thread_id=event.thread_id, byr_hist=byr_hist)

    def _check_complete(self, event):
        return False

    def make_thread(self, priority):
        return RewardThread(priority=priority, thread_id=self.thread_counter,
                            buyer=SimulatedBuyer(model=self.buyer),
                            seller=SimulatedSeller(model=self.seller))




import numpy as np
from rlenv.events.RewardThread import RewardThread
from rlenv.environments.EbayEnvironment import EbayEnvironment
from constants import MONTH
from featnames import START_TIME, TIME_FEATS


class SimulatorEnvironment(EbayEnvironment):
    def __init__(self, **kwargs):
        super(SimulatorEnvironment, self).__init__(
            kwargs['arrival'], kwargs['verbose'])

        # save parameters to self
        self.buyer = kwargs['buyer']
        self.seller = kwargs['seller']
        self.x_lstg = kwargs['x_lstg']
        self.lookup = kwargs['lookup']
        self.recorder = kwargs['recorder']

        # end time
        self.end_time = self.lookup[START_TIME] + MONTH
        self.thread_counter = 0

    def reset(self):
        super(SimulatorEnvironment, self).reset()
        self.recorder.reset_sim()
        if self.verbose:
            print('Simulation {}'.format(self.recorder.sim))

    def run(self):
        """
        Runs a simulation of a single lstg until sale or expiration

        :return: a 3-tuple of (bool, float, int) giving whether the listing sells,
        the amount it sells for if it sells, and the amount of time it took to sell
        """
        super(SimulatorEnvironment, self).run()
        return self.outcome

    def _record(self, event, byr_hist=None, censored=False):
        """
        Add record of offer or thread to Recorder
        :param rlenv.events.ThreadSources event: event containing most recent offer
        """
        if byr_hist is None:
            if not censored:
                time_feats = self.time_feats.get_feats(thread_id=event.thread_id,
                                                       time=event.priority)
            else:
                time_feats = np.zeros(len(TIME_FEATS))
            self.recorder.add_offer(event=event, time_feats=time_feats, censored=censored)
        else:
            self.recorder.start_thread(thread_id=event.thread_id, byr_hist=byr_hist,
                                       time=event.priority)

    def _is_agent_turn(self, event):
        return False

    def make_thread(self, priority):
        return RewardThread(priority=priority, thread_id=self.thread_counter,
                            intervals=self.intervals, buyer=self.buyer, seller=self.seller)




import numpy as np
from rlenv.environments.EbayEnvironment import EbayEnvironment
from constants import MONTH
from featnames import START_TIME, TIME_FEATS, X_LSTG, LOOKUP, P_ARRIVAL


class SimulatorEnvironment(EbayEnvironment):
    def __init__(self, **kwargs):
        super().__init__(kwargs)

        self.x_lstg = kwargs[X_LSTG]
        self.lookup = kwargs[LOOKUP]
        self.p_arrival = kwargs[P_ARRIVAL]
        self.recorder = kwargs['recorder']

        # end time
        self.start_time = self.lookup[START_TIME]
        self.end_time = self.start_time + MONTH

    def reset(self):
        super().reset()
        if self.recorder is not None:
            self.recorder.reset_sim()
        if self.verbose:
            print('Simulation {}'.format(self.recorder.sim))

    def run(self):
        """
        Runs a simulation of a single lstg until sale or expiration

        :return: a 3-tuple of (bool, float, int) giving whether the listing sells,
        the amount it sells for if it sells, and the amount of time it took to sell
        """
        super().run()
        return self.outcome

    def record(self, event, byr_hist=None, censored=False):
        """
        Add record of offer or thread to Recorder
        :param censored:
        :param byr_hist:
        :param rlenv.events.Thread.Thread event: event containing most recent offer
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

    def is_agent_turn(self, event):
        return False




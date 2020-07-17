import numpy as np
from rlenv.environments.EbayEnvironment import EbayEnvironment
from featnames import TIME_FEATS


class SimulatorEnvironment(EbayEnvironment):
    def __init__(self, **kwargs):
        super().__init__(kwargs)

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

    def is_agent_turn(self, event):
        return False




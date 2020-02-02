from rlenv.environments.SimulatorEnvironment import SimulatorEnvironment
from rlenv.test.LstgLog import LstgLog


class TestEnvironment(SimulatorEnvironment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lstg_log = kwargs['log']  # type: LstgLog

    def record(self, event, byr_hist=None, censored=None):
        """Record nothing"""
        pass

    def get_con(self, event=None):
        con = self.lstg_log.get_con(event)
        return con

    def get_inter_arrival(self, input_dict=None, time=None):
        return self.lstg_log.get_inter_arrival(input_dict=input_dict, time=time,
                                               thread_id=self.thread_counter)

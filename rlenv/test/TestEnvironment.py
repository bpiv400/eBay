from rlenv.environments.SimulatorEnvironment import SimulatorEnvironment
from rlenv.test.LstgLog import LstgLog


class TestEnvironment(SimulatorEnvironment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lstg_log = kwargs['log']  # type: LstgLog

    def get_arrival(self, input_dict=None, time=None, first=None, intervals=None):
        return self.lstg_log.get_inter_arrival(input_dict=input_dict, time=time,
                                               thread_id=self.thread_counter)

    def get_hist(self, input_dict=None, time=None, thread_id=None):
        return self.lstg_log.get_hist(thread_id=thread_id, time=time,
                                      input_dict=input_dict)

    def get_con(self, input_dict=None, time=None, thread_id=None, turn=None):
        con = self.lstg_log.get_con(input_dict=input_dict, thread_id=thread_id,
                                    time=time, turn=turn)
        return con

    def get_msg(self, input_dict=None, time=None, thread_id=None, turn=None):
        return self.lstg_log.get_msg(input_dict=input_dict, thread_id=thread_id,
                                     time=time, turn=turn)

    def get_delay(self, input_dict=None, turn=None, thread_id=None,
                  time=None, delay_type=None):
        return self.lstg_log.get_delay(thread_id=thread_id, turn=turn,
                                       input_dict=input_dict, time=time)


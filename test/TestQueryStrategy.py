from QueryStrategy import QueryStrategy
from test.LstgLog import LstgLog


class TestQueryStrategy(QueryStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lstg_log = kwargs['log']  # type: LstgLog

    def update_log(self, log):
        self.lstg_log = log

    def get_arrival(self, *args, **kwargs):
        return self.lstg_log.get_inter_arrival(input_dict=input_dict, time=time,
                                               thread_id=self.thread_counter)

    def get_hist(self, *args, **kwargs):
        return self.lstg_log.get_hist(thread_id=thread_id, time=time,
                                      input_dict=input_dict)

    def get_con(self, *args, **kwargs):
        con = self.lstg_log.get_con(input_dict=input_dict, thread_id=thread_id,
                                    time=time, turn=turn)
        return con

    def get_msg(self, *args, **kwargs):
        return self.lstg_log.get_msg(input_dict=input_dict, thread_id=thread_id,
                                     time=time, turn=turn)

    def get_delay(self, *args, **kwargs):
        return self.lstg_log.get_delay(thread_id=thread_id, turn=turn,
                                       input_dict=input_dict, time=time)


from env.QueryStrategy import QueryStrategy


class TestQueryStrategy(QueryStrategy):
    def __init__(self):
        super().__init__()
        self.lstg_log = None

    def update_log(self, log):
        self.lstg_log = log

    def get_first_arrival(self, *args, **kwargs):
        return self.lstg_log.get_inter_arrival(time=kwargs['time'], thread_id=1)

    def get_inter_arrival(self, *args, **kwargs):
        return self.lstg_log.get_inter_arrival(input_dict=kwargs['input_dict'],
                                               time=kwargs['time'],
                                               thread_id=kwargs['thread_id'])

    def get_hist(self, *args, **kwargs):
        return self.lstg_log.get_hist(thread_id=kwargs['thread_id'],
                                      time=kwargs['time'],
                                      input_dict=kwargs['input_dict'])

    def get_con(self, *args, **kwargs):
        return self.lstg_log.get_con(thread_id=kwargs['thread_id'],
                                     time=kwargs['time'],
                                     input_dict=kwargs['input_dict'],
                                     turn=kwargs['turn'])

    def get_msg(self, *args, **kwargs):
        return self.lstg_log.get_msg(thread_id=kwargs['thread_id'],
                                     time=kwargs['time'],
                                     input_dict=kwargs['input_dict'],
                                     turn=kwargs['turn'])

    def get_delay(self, *args, **kwargs):
        return self.lstg_log.get_delay(thread_id=kwargs['thread_id'],
                                       time=kwargs['time'],
                                       input_dict=kwargs['input_dict'],
                                       turn=kwargs['turn'])

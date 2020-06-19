import pandas as pd
from featnames import START_TIME, MONTHS_SINCE_LSTG, BYR_HIST, CON, AUTO
from constants import (MONTH, MODELS, FIRST_ARRIVAL_MODEL,
                       BYR_HIST_MODEL, INTERARRIVAL_MODEL, OFFER_MODELS)
from rlenv.util import populate_test_model_inputs
from test.ArrivalLog import ArrivalLog
from test.ThreadLog import ThreadLog


class LstgLog:

    def __init__(self, params=None):
        """
        :param params: dict
        """
        self.lstg = params['lstg']
        self.lookup = params['lookup']
        params = LstgLog.subset_params(params)
        self.arrivals = self.generate_arrival_logs(params)
        self.threads = self.generate_thread_logs(params)

    @property
    def has_arrivals(self):
        return not self.arrivals[1].censored

    def generate_thread_logs(self, params):
        thread_logs = dict()
        for thread_id, arrival_log in self.arrivals.items():
            if not arrival_log.censored:
                print('Thread id: {}'.format(thread_id))
                thread_logs[thread_id] = self.generate_thread_log(thread_id=thread_id, params=params)
        return thread_logs

    def generate_arrival_logs(self, params):
        arrival_logs = dict()
        if params['x_thread'] is None:
            censored = self.generate_censored_arrival(params=params, thread_id=1)
            arrival_logs[1] = censored
        else:
            num_arrivals = len(params['x_thread'].index)
            for i in range(1, num_arrivals + 1):
                curr_arrival = self.generate_arrival_log(params=params,
                                                         thread_id=i)
                arrival_logs[i] = curr_arrival

            if not self.check_bin(params=params, thread_id=num_arrivals):
                censored = self.generate_censored_arrival(params=params, thread_id=num_arrivals + 1)
                arrival_logs[num_arrivals + 1] = censored
            else:
                print('has bin')
        return arrival_logs

    def generate_censored_arrival(self, params=None, thread_id=None):
        if thread_id == 1:
            model = FIRST_ARRIVAL_MODEL
            value = None
        else:
            model = INTERARRIVAL_MODEL
            value = thread_id
        check_time = self.arrival_check_time(params=params, thread_id=thread_id)
        full_arrival_inputs = params['inputs'][model]
        # print(full_arrival_inputs)
        arrival_inputs = populate_test_model_inputs(full_inputs=full_arrival_inputs,
                                                    value=value)
        time = self.lookup[START_TIME] + MONTH
        return ArrivalLog(check_time=check_time, arrival_inputs=arrival_inputs, time=time,
                          first_arrival=thread_id == 1)

    def arrival_check_time(self, params=None, thread_id=None):
        if thread_id == 1:
            check_time = self.lookup[START_TIME]
        else:
            check_time = int(params['x_thread'].loc[thread_id - 1, MONTHS_SINCE_LSTG] * MONTH)
            check_time += self.lookup[START_TIME]
        return check_time

    def generate_arrival_log(self, params=None, thread_id=None):
        if thread_id == 1:
            model = FIRST_ARRIVAL_MODEL
            value = None
        else:
            model = INTERARRIVAL_MODEL
            value = thread_id
        check_time = self.arrival_check_time(params=params, thread_id=thread_id)
        time = int(params['x_thread'].loc[thread_id, MONTHS_SINCE_LSTG] * MONTH)
        time += self.lookup[START_TIME]
        hist = params['x_thread'].loc[thread_id, BYR_HIST] / 10
        full_arrival_inputs = params['inputs'][model]
        full_hist_inputs = params['inputs'][BYR_HIST_MODEL]
        print('value: {}'.format(value))
        arrival_inputs = populate_test_model_inputs(full_inputs=full_arrival_inputs,
                                                    value=value)
        hist_inputs = populate_test_model_inputs(full_inputs=full_hist_inputs,
                                                 value=thread_id)
        return ArrivalLog(hist=hist, time=time, arrival_inputs=arrival_inputs,
                          hist_inputs=hist_inputs, check_time=check_time,
                          first_arrival=thread_id == 1)

    def generate_thread_log(self, thread_id=None, params=None):
        thread_params = dict()
        thread_params['x_offer'] = params['x_offer'].xs(thread_id, level='thread', drop_level=True)
        thread_params['inputs'] = LstgLog.subset_inputs(models=OFFER_MODELS, input_data=params['inputs'],
                                                        value=thread_id, level='thread')
        return ThreadLog(params=thread_params, arrival_time=self.arrivals[thread_id].time)

    def get_con(self, thread_id=None, turn=None, input_dict=None, time=None):
        """
        :return: np.float
        """
        con = self.threads[thread_id].get_con(turn=turn, time=time, input_dict=input_dict)
        return con

    def get_msg(self, thread_id=None, turn=None, input_dict=None, time=None):
        """
        :return: np.float
        """
        msg = self.threads[thread_id].get_msg(turn=turn, time=time, input_dict=input_dict)
        if msg:
            return 1.0
        else:
            return 0.0

    def get_delay(self, thread_id=None, turn=None, input_dict=None, time=None):
        """
        :return: int
        """
        delay = self.threads[thread_id].get_delay(turn=turn, time=time, input_dict=input_dict)
        if delay == MONTH:
            return self.lookup[START_TIME] + MONTH - time
        else:
            return delay

    def get_inter_arrival(self, thread_id=None, input_dict=None, time=None):
        return self.arrivals[thread_id].get_inter_arrival(check_time=time, input_dict=input_dict)

    def get_hist(self, thread_id=None, input_dict=None, time=None):
        return self.arrivals[thread_id].get_hist(check_time=time, input_dict=input_dict)

    @staticmethod
    def check_bin(params=None, thread_id=None):
        thread1 = params['x_offer'].xs(thread_id, level='thread')
        if len(thread1.index) == 1:
            assert thread1.loc[1, CON] == 1
            return True
        elif len(thread1.index) == 2:
            return thread1.loc[2, AUTO] and (thread1.loc[2, CON] == 1)

    @staticmethod
    def subset_params(params=None):
        params = params.copy()
        params['x_offer'] = LstgLog.subset_df(df=params['x_offer'],
                                              lstg=params['lstg'])
        params['x_thread'] = LstgLog.subset_df(df=params['x_thread'],
                                               lstg=params['lstg'])
        params['inputs'] = LstgLog.subset_inputs(input_data=params['inputs'], models=MODELS,
                                                 level='lstg', value=params['lstg'])
        # print(params['inputs']['arrival']['lstg'])
        return params

    @staticmethod
    def subset_df(df=None, lstg=None):
        """
        Subsets an arbitrary dataframe to only contain rows for the given
        lstg
        :param df: pd.DataFrame
        :param lstg: integer giving lstg
        :return: pd.Series or pd.DataFrame
        """
        if isinstance(df.index, pd.MultiIndex):
            lstgs = df.index.unique(level='lstg')
            if lstg in lstgs:
                return df.xs(lstg, level='lstg', drop_level=True)
        else:
            if lstg in df.index:
                return df.loc[lstg, :]
        return None

    @staticmethod
    def subset_inputs(models=None, input_data=None, value=None, level=None):
        inputs = dict()
        for model in models:
            inputs[model] = dict()
            index_is_cached = False
            curr_index = None
            for input_group, feats_df in input_data[model].items():
                if feats_df is not None:
                    multi_index = isinstance(feats_df.index, pd.MultiIndex)
                else:
                    multi_index = False
                if not index_is_cached:
                    if multi_index:
                        contains_value = feats_df is not None and\
                                         (value in feats_df.index.unique(level=level))
                    else:
                        contains_value = feats_df is not None and\
                                         (value in feats_df.index)
                    if contains_value:
                        if multi_index:
                            full_lstgs = feats_df.index.get_level_values(level)
                        else:
                            full_lstgs = feats_df.index
                        curr_index = full_lstgs == value
                    else:
                        curr_index = None
                    index_is_cached = True
                if curr_index is not None:
                    subset = feats_df.loc[curr_index, :]
                    if multi_index:
                        subset.index = subset.index.droplevel(level=level)
                else:
                    subset = None
                inputs[model][input_group] = subset
        return inputs

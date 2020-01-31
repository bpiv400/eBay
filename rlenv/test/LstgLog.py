import pandas as pd
import torch
from featnames import START_TIME, MONTHS_SINCE_LSTG, BYR_HIST
from rlenv.test.ThreadLog import ThreadLog
from rlenv.env_consts import MODELS, ARRIVAL_MODEL, BYR_HIST_MODEL
from constants import MONTH

class LstgLog:

    def __init__(self, params=None):
        """
        # TODO: Update
        :param params: A format decided upon by Etan and I in meeting
        """
        self.lstg = params['lstg']
        self.lookup = params['lookup']
        LstgLog.subset_params(params)
        self.arrivals = self.generate_arrival_logs(params)
        self.threads = self.generate_thread_logs(params)
        self.x_lstg = None # TODO: Create a dictionary containing all the x_lstg components that are common
        # among all models / turns (e.g. all but x_lstg)

    @property
    def has_arrivals(self):
        return len(self.arrivals) > 0

    def generate_arrival_logs(self, params):
        arrival_logs = dict()
        if params['x_thread'] is None:
            censored = self.generate_censored_arrival(params=params, thread_id=1)
            arrival_logs[1] = censored
        else:
            for i in range(1, len(params['x_thread'].index) + 1):
                curr_arrival = self.generate_arrival_log(params=params,
                                                         thread_id=i)
                arrival_logs[i] = curr_arrival

            if not self.check_bin(params=params, thread_id=i):
                censored = self.generate_censored_arrival(params=params, thread_id=i+1)

    def add_censored_arrival(self, params=None, thread_id=None):
        check_time = self.arrival_check_time(params=params, thread_id=thread_id)
        full_arrival_inputs = params['inputs'][ARRIVAL_MODEL]
        arrival_inputs = self.populate_thread_inputs(full_inputs=full_arrival_inputs,
                                                     thread_id=thread_id)
        time = self.lookup[START_TIME] + MONTH
        return ArrivalLog(check_time=check_time, arrival_inputs=arrival_inputs)

    def arrival_check_time(self, params=None, thread_id=None):
        if thread_id == 1:
            check_time = self.lookup[START_TIME]
        else:
            check_time = int(params['x_thread'].loc[thread_id - 1, MONTHS_SINCE_LSTG] * MONTH)
        return check_time

    def generate_arrival_log(self, params=None, thread_id=None):
        check_time = self.arrival_check_time(params=params, thread_id=thread_id)
        time = int(params['x_thread'].loc[thread_id, MONTHS_SINCE_LSTG] * MONTH)
        time += self.lookup[START_TIME]
        hist = params['x_thread'].loc[thread_id, BYR_HIST]
        full_arrival_inputs = params['inputs'][ARRIVAL_MODEL]
        full_hist_inputs = params['inputs'][BYR_HIST_MODEL]
        arrival_inputs = self.populate_thread_inputs(full_inputs=full_arrival_inputs,
                                                     thread_id=thread_id)
        hist_inputs = self.populate_thread_inputs(full_inputs=full_hist_inputs,
                                                  thread_id=thread_id)
        return ArrivalLog(hist=hist, time=time, arrival_inputs=arrival_inputs,
                          hist_inputs=hist_inputs, check_time=check_time)

    @staticmethod
    def populate_thread_inputs(full_inputs=None, thread_id=None):
        inputs = dict()
        for feat_set_name, feat_df in full_inputs.items():
            curr_set = full_inputs[feat_set_name].loc[thread_id, :]
            curr_set = torch.from_numpy(curr_set.values).float().unsqueeze(0)
            inputs[feat_set_name] = curr_set
        return inputs


    def generate_thread_log(self, thread=None):
        print(self.params)
        # TODO: Creates a ThreadLog for the given thread containing outcome data for each turn
        # TODO: of the the thread & model input data for instance a model is run
        # subset params
        return dict()

    def get_con(self, event=None):
        """

        :param rlenv.Events.Thread.Thread event:
        :return: np.float
        """
        con = self.threads[event.thread_id].get_con(event=event, x_lstg=self.x_lstg)

    @staticmethod
    def check_bin(params=None, thread_id=None):
        first_offer = params['x_offer'].xs(thread_id, level='thread', drop_level=True).loc[1, :]
        return first_offer == 1

    @staticmethod
    def subset_params(params=None):
        params['x_offer'] = LstgLog.subset_df(df=params['x_offer'],
                                              lstg=params['lstg'])
        params['x_thread'] = LstgLog.subset_df(df=params['x_thread'],
                                              lstg=params['lstg'])
        params['inputs'] = LstgLog.subset_inputs(input_data=params['inputs'],
                                                 lstg=params['lstg'])

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
    def subset_inputs(input_data=None, lstg=None):
        inputs = dict()
        for model in MODELS:
            inputs[model] = dict()
            index_is_cached = False
            curr_index = None
            for input_group, feats_df in input_data[model].items():
                if not index_is_cached:
                    if lstg in feats_df.index.unique(level='lstg'):
                        full_lstgs = feats_df.index.get_level_values('lstg')
                        curr_index = full_lstgs == lstg
                    else:
                        curr_index = None
                    index_is_cached = True
                if curr_index is None:
                    subset = feats_df.loc[curr_index, :]
                    subset.index = subset.index.droplevel(level='lstg')
                else:
                    subset = None
                inputs[model][input_group] = subset
        return input_data



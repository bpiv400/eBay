import pandas as pd
from agent.util import get_agent_name
from featnames import START_TIME, MONTHS_SINCE_LSTG, BYR_HIST, CON, AUTO
from constants import (MONTH, MODELS, FIRST_ARRIVAL_MODEL, DAY,
                       BYR_HIST_MODEL, INTERARRIVAL_MODEL, OFFER_MODELS)
from rlenv.util import populate_test_model_inputs
from test.AgentLog import AgentLog, ActionLog
from test.ArrivalLog import ArrivalLog
from test.ThreadLog import ThreadLog
from utils import init_optional_arg


class LstgLog:

    def __init__(self, params=None):
        """
        :param params: dict
        """
        init_optional_arg(kwargs=params,
                          name='agent_params',
                          default=None)
        self.lstg = params['lstg']
        self.lookup = params['lookup']
        self.agent = params['agent_params'] is not None
        self.agent_params = params['agent_params']
        params = LstgLog.subset_params(params)
        self.arrivals = self.generate_arrival_logs(params)
        self.threads = self.generate_thread_logs(params)
        self.agent_log = self.generate_agent_log(params)
        if self.agent and self.agent_params['byr']:
            self.skip_agent_arrival()

    @property
    def byr(self):
        if self.agent:
            return self.agent_params['byr']
        else:
            raise RuntimeError("byr not defined")

    @property
    def agent_thread(self):
        if self.agent:
            if 'thread_id' in self.agent_params:
                return self.agent_thread
            else:
                return None
        else:
            raise RuntimeError('agent thread not defined')

    @property
    def delay(self):
        if self.agent:
            return self.agent_params['delay']
        else:
            raise RuntimeError('Delay not defined')

    def generate_agent_log(self, params):
        # if the agent is a buyer
        agent_log = AgentLog(byr=self.byr,
                             delay=self.delay)
        full_inputs = params['inputs'][agent_log.model_name]
        if self.byr:
            byr_arrival = self.arrivals[self.agent_thread]
            # find the floor of the number of days that have passed since the start of the
            days = int(byr_arrival.time - self.lookup[START_TIME] / DAY)
            for i in range(days):
                action_index = (self.agent_thread, 1, i)
                input_dict = populate_test_model_inputs(full_inputs=full_inputs,
                                                        value=action_index)
                time = (i * DAY) / MONTH
                log = ActionLog(input_dict=input_dict, months=time, con=0,
                                thread_id=self.agent_thread, turn=1)
                agent_log.push_action(action=log)

            # set interarrival time to account for query of first arrival model
            interarrival_time = byr_arrival.time - (days * DAY + self.lookup[START_TIME])
            self.agent_log.set_interarrival_time(interarrival_time=interarrival_time)


            time = (days * DAY) / MONTH
            first_turn_index = (self.agent_thread, 1, days)
            con = byr_thread.agent_con(turn=1)
            input_dict = populate_test_model_inputs(full_inputs=full_inputs, value=first_turn_index)
            first_offer = ActionLog
            # for the last day, insert an action that corresponds to the offer
            # (extract from thread)
            # store the difference between the check time on that last day and the offer
            # to feed arrival model when it queries for the delay (could verify intervals here)
        # insert remaining offers into queue


    def skip_agent_arrival(self):
        byr_arrival = self.arrivals[self.agent_thread]
        if byr_arrival.censored:
            raise RuntimeError("Agent buyer arrival must not be censored")
        # if this is the first arrival
            # if this is a buy it now event (i.e. there's no censored next arrival)
                # replace the interarrival with the end of the lstg
                # censor the arrival
            # if this is not a buy it now event (i.e there's an arrival next (censored or not)
                # replace the interarrival time with the difference between the check time
                # and the time of the next arrival
        # if there is an arrival before this one
            # if this is a buy it now event (i.e. there's no censored next arrival)
                # replace the interarrival time of the previous arrival with the end of the lstg
                # censored the previous arrival
            # if this is not a buy it now event (i.e. there's a next arrival)
                # replace the interarrival time of the previous arrival with the
                # difference between the time of the next arrival and the check time
                # of the previous arrival

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
                          first_arrival=thread_id == 1,
                          agent=self.is_agent_arrival(thread_id=thread_id))

    def is_agent_arrival(self, thread_id=None):
        if self.agent:
            if self.byr:
                return thread_id == self.agent_thread
        return False

    def is_agent_thread(self, thread_id=None):
        if self.agent:
            if self.byr:
                return thread_id == self.agent_thread
            else:
                return True
        else:
            return False

    def generate_thread_log(self, thread_id=None, params=None):
        thread_params = dict()
        thread_params['x_offer'] = params['x_offer'].xs(thread_id, level='thread', drop_level=True)
        thread_params['inputs'] = LstgLog.subset_inputs(models=OFFER_MODELS, input_data=params['inputs'],
                                                        value=thread_id, level='thread')
        agent_thread = self.is_agent_thread(thread_id=thread_id)
        if agent_thread:
            agent_buyer = self.byr
        else:
            agent_buyer = False
        return ThreadLog(params=thread_params, arrival_time=self.arrivals[thread_id].time,
                         agent=agent_thread, agent_buyer=agent_buyer)

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
        params['x_offer'] = subset_df(df=params['x_offer'],\
                                      lstg=params['lstg'])
        params['x_thread'] = subset_df(df=params['x_thread'],
                                       lstg=params['lstg'])
        params['inputs'] = subset_inputs(input_data=params['inputs'], models=MODELS,
                                         level='lstg', value=params['lstg'])
        # print(params['inputs']['arrival']['lstg'])
        return params

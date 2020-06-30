import pandas as pd
from agent.util import get_agent_name
from featnames import START_TIME, MONTHS_SINCE_LSTG, BYR_HIST, CON, AUTO, LSTG
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
        self.agent_log = self.generate_agent_log(params) # revisit this
        if self.agent and self.byr:
            self.translator = ThreadTranslator(agent_thread=self.agent_thread,
                                               arrivals=self.arrivals)
            self.update_arrival_time()

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

    def record_agent_arrivals(self, full_inputs=None, agent_log=None):
        byr_arrival = self.arrivals[self.agent_thread]
        # find the floor of the number of days that have passed since the start of the
        days = int(byr_arrival.time - self.lookup[START_TIME] / DAY)
        for i in range(days):
            action_index = (self.agent_thread, 1, i)
            input_dict = populate_test_model_inputs(full_inputs=full_inputs,
                                                    value=action_index,
                                                    agent=True,
                                                    agent_byr=True)
            time = (i * DAY) / MONTH
            log = ActionLog(input_dict=input_dict, months=time, con=0,
                            thread_id=self.agent_thread, turn=1)
            agent_log.push_action(action=log)
        return days

    def record_buyer_first_turn(self, full_inputs=None, agent_log=None,
                                days=None, agent_turns=None):
        con = agent_turns[1].agent_con()
        time = (days * DAY) / MONTH
        first_turn_index = (self.agent_thread, 1, days)
        input_dict = populate_test_model_inputs(full_inputs=full_inputs, value=first_turn_index,
                                                agent=True, agent_byr=self.byr)
        first_offer = ActionLog(input_dict=input_dict, months=time, con=con,
                                thread_id=self.agent_thread, turn=1)
        agent_log.push_action(action=first_offer)
        # add remaining turns
        del agent_turns[1]

    def record_agent_thread(self, turns=None, agent_log=None, thread_id=None, full_inputs=None):
        for turn_number, turn_log in turns.items():
            time = turn_log.agent_time(delay=self.delay)
            months = (time - self.lookup[START_TIME]) / MONTH
            con = turn_log.agent_con()
            if self.byr:
                index = (thread_id, turn_number, 0)
            else:
                index = (thread_id, turn_number)
            if turn_log.is_censored:
                input_dict = None
            else:
                input_dict = populate_test_model_inputs(full_inputs=full_inputs, value=index,
                                                       agent=True, agent_byr=self.byr)
            action = ActionLog(con=turn_log.agent_con(), censored=turn_log.is_censored,
                               months=months, input_dict=input_dict, thread_id=thread_id,
                               turn=turn_number)
            agent_log.push_action(action=action)

    def generate_buyer_log(self, full_inputs=None, agent_log=None):
        days = self.record_agent_arrivals(full_inputs=full_inputs, agent_log=agent_log)
        # add first turn
        agent_turns = self.threads[self.agent_thread].get_agent_turns(delay=self.delay)
        self.record_buyer_first_turn(full_inputs=full_inputs, agent_log=agent_log,
                                     days=days, agent_turns=agent_turns)
        # record remaining threads
        self.record_agent_thread(turns=agent_turns, agent_log=agent_log,
                                 thread_id=self.agent_thread, full_inputs=full_inputs)

    def generate_agent_log(self, params):
        # if the agent is a buyer
        agent_log = AgentLog(byr=self.byr,
                             delay=self.delay)
        full_inputs = params['inputs'][agent_log.model_name]
        if self.byr:
            self.generate_buyer_log(full_inputs=full_inputs, agent_log=agent_log)
        else:
            self.generate_seller_log(full_inputs=full_inputs, agent_log=agent_log)
        return agent_log

    def generate_seller_log(self, full_inputs=None, agent_log=None):
        for thread_id, thread_log in self.threads.items():
            agent_turns = thread_log.get_agent_turns(delay=self.delay)
            if len(agent_turns) != 0:
                self.record_agent_thread(agent_log=agent_log, full_inputs=full_inputs,
                                         thread_id=thread_id, turns=agent_turns)

    def update_arrival_time(self):
        if (self.agent_thread + 1) in self.arrivals:
            target_time = self.arrivals[self.agent_thread + 1].time
        else:
            target_time = self.lookup[START_TIME] + MONTH
        target_censored = target_time == (self.lookup[START_TIME] + MONTH)
        self.arrivals[self.agent_thread].time = target_time
        self.arrivals[self.agent_thread].censored = target_censored

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
        true_id = self.translate_thread(thread_id=thread_id)
        con = self.threads[true_id].get_con(turn=turn, time=time, input_dict=input_dict)
        return con

    def get_msg(self, thread_id=None, turn=None, input_dict=None, time=None):
        """
        :return: np.float
        """
        true_id = self.translate_thread(thread_id=thread_id)
        msg = self.threads[true_id].get_msg(turn=turn, time=time, input_dict=input_dict)
        if msg:
            return 1.0
        else:
            return 0.0

    def get_delay(self, thread_id=None, turn=None, input_dict=None, time=None):
        """
        :return: int
        """
        true_id = self.translate_thread(thread_id=thread_id)
        delay = self.threads[true_id].get_delay(turn=turn, time=time, input_dict=input_dict)
        if delay == MONTH:
            return self.lookup[START_TIME] + MONTH - time
        else:
            return delay

    def get_inter_arrival(self, thread_id=None, input_dict=None, time=None, agent=False):
        if agent:
            return self.translator.get_agent_arrival(check_time=time, thread_id=thread_id)
        else:
            true_id = self.translate_arrival(thread_id=thread_id)
            return self.arrivals[true_id].get_inter_arrival(check_time=time,
                                                            input_dict=input_dict)

    def get_hist(self, thread_id=None, input_dict=None, time=None):
        return self.arrivals[thread_id].get_hist(check_time=time, input_dict=input_dict)

    def translate_thread(self, thread_id=None):
        if self.translator is None:
            return thread_id
        else:
            return self.translator.translate_thread(thread_id)

    def translate_arrival(self, thread_id=None):
        if self.translator is None:
            return thread_id
        else:
            return self.translator.translate_arrival(thread_id)

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



class ThreadTranslator:
    def __init__(self, arrivals=None, agent_thread=None, params=None):
        self.agent_thread = agent_thread
        # boolean for whether the agent is the first thread
        self.agent_first = self.agent_thread == 1
        # boolean for whether the agent is the last thread
        self.agent_last = self.agent_thread == len(arrivals)
        # time when the buyer agent model is queried and produces a first offer
        self.agent_check_time = self.get_rl_check_time(params=params)
        # time when the buyer agent model executes the first offer
        self.agent_arrival_time = arrivals[self.agent_thread].time
        # number of seconds the first arrival model should delay for when
        # queried for arrival time of the buyer's first offer
        self.agent_interarrival_time = self.agent_arrival_time - self.agent_check_time
        # see get_thread_l for description
        self.thread_l = self.get_thread_l(arrivals=arrivals)
        if self.thread_l is not None:
            # boolean for whether thread l arrival time is after agent arrival time
            self.l_after_agent = arrivals[self.thread_l].time > self.agent_arrival_time
            # boolean for whether thread_l is censored
            self.l_censored = arrivals[self.thread_l].censored
            # counter of threads with arrival time >= rl_check_time
            # j should be < 0 when l_censored
            self.j = self.agent_thread - self.thread_l
        else:
            self.l_censored, self.l_after_agent = None, None
            self.j = None
        # boolean for whether the id will be queried twice
        self.query_twice = self.get_query_twice()
        self.agent_env_id = self.get_agent_env_id()
        self.hidden_arrival = self.get_hidden_arrival()
        self.arrival_translator = self.make_arrival_translator()
        self.thread_translator = self.make_thread_translator()
        # flag for whether the agent_env_id has been queried in the arrival
        # process
        self.did_query = False

    def get_query_twice(self):
        if self.thread_l is not None:
            return self.l_censored
        else:
            return self.agent_last


    def get_agent_env_id(self):
        if self.agent_first:
            if self.query_twice:
                return 1
            else:
                return 2
        else:
            if self.thread_l is not None:
                if self.l_after_agent:
                    if self.l_censored:
                        return self.thread_l - 1 # query_twice
                    else:
                        return self.thread_l
                else:
                    return self.thread_l + 1
            else:
                return self.agent_thread

    def get_hidden_arrival(self):
        if self.query_twice:
             return None
        else:
            return self.agent_thread


    def get_thread_l(self, arrivals=None):
        """
        first thread where time >= rl_check_time,
        None if rl thread is the last thread (meaning there are no censored
        arrivals) and there are no arrivals after the rl check time,
        before the rl arrival time
        :param arrivals: dictionary
        :return:
        """
        after_rl_check = list()
        for thread_id, arrival_log in arrivals.items():
            if arrival_log.time >= self.agent_check_time and \
                    thread_id != self.agent_thread:
                after_rl_check.append(thread_id)
        if len(after_rl_check) == 0:
            assert self.agent_last
            return None
        else:
            return min(after_rl_check)


    def get_rl_check_time(self, params=None):
        df = params['inputs'][get_agent_name(byr=True,
                                             delay=True,
                                             policy=True)][LSTG]
        df = df.xs(key=self.agent_thread, level='thread',
                   drop_level=True)
        df = df.xs(key=1, level='index', drop_level=True)
        day = df.index.max()
        return (day * DAY) + params['lookup'][START_TIME]


    def make_arrival_translator(self):
        """
        Dictionary that translates the env id of arrival queries to
        their original id in the recorded trajectories
        :return: dict
        """
        translator = dict()
        if self.thread_l is not None and self.j >= 1:
            for env_id in range(self.thread_l + 2, self.thread_l + self.j + 2):
                translator[env_id] = env_id - 1
        # leaving comments to track logic
        # at least one thread with time > rl_arrival_time
        # all threads have their true arrival log
        # elif self.thread_l is not None:
        #    return translator
        # no arrivals after agent_check_time, agent is last arrival
        #else:
        #    return translator
        return translator

    def make_thread_translator(self):
        translator = dict()
        if self.thread_l is not None:
            if self.j >= 1:
                translator[self.thread_l + 1] = self.thread_l + self.j
                for env_id in range(self.thread_l + 2, self.thread_l + self.j + 1):
                    translator[env_id] = env_id - 1
            elif not self.l_censored:
                    translator[self.thread_l - 1] = self.thread_l
                    translator[self.thread_l] = self.thread_l - 1
        return translator

    def translate_thread(self, env_id):
        if env_id in self.thread_translator:
            return self.thread_translator[env_id]
        else:
            return env_id

    def get_agent_arrival(self, thread_id=None, check_time=None):
        assert thread_id == self.agent_env_id
        assert check_time == self.agent_check_time
        return self.agent_interarrival_time

    def translate_arrival(self, env_id):
        if env_id == self.hidden_arrival:
            raise RuntimeError("Should not query arrival for %s" %
                               env_id)
        else:
            if env_id in self.arrival_translator:
                return self.arrival_translator[env_id]
            else:
                return env_id




from featnames import START_TIME, MONTHS_SINCE_LSTG, BYR_HIST, CON, AUTO
from constants import (MONTH, DAY, BYR_HIST_MODEL,
                       INTERARRIVAL_MODEL, OFFER_MODELS)
from testing.AgentLog import AgentLog
from testing.ActionLog import ActionLog
from testing.ArrivalLog import ArrivalLog
from testing.ThreadLog import ThreadLog
from testing.ThreadTranslator import ThreadTranslator
from testing.util import subset_inputs, populate_test_model_inputs
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
        self.verbose = params['verbose']
        self.arrivals = self.generate_arrival_logs(params)
        self.threads = self.generate_thread_logs(params)
        if self.agent and self.byr:
            self.translator = ThreadTranslator(agent_thread=self.agent_thread,
                                               arrivals=self.arrivals,
                                               params=params)
            if self.verbose:
                self.translator.print_translator()
        else:
            self.translator = None
        self.agent_log = self.generate_agent_log(params)
        self.update_arrival_time()

    @property
    def byr(self):
        if self.agent:
            return self.agent_params['byr']
        else:
            return False

    @property
    def agent_thread(self):
        if self.agent:
            if 'thread_id' in self.agent_params:
                return self.agent_params['thread_id']
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
        days = int((byr_arrival.time - self.lookup[START_TIME]) / DAY)
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
                                thread_id=self.translate_thread(self.agent_thread),
                                turn=1)
        agent_log.push_action(action=first_offer)
        # add remaining turns
        del agent_turns[1]

    def record_agent_thread(self, turns=None, agent_log=None, thread_id=None, full_inputs=None):
        for turn_number, turn_log in turns.items():
            time = turn_log.agent_time()
            months = (time - self.lookup[START_TIME]) / MONTH
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
                               months=months, input_dict=input_dict,
                               thread_id=self.translate_thread(self.agent_thread),
                               turn=turn_number)
            agent_log.push_action(action=action)

    def generate_buyer_log(self, full_inputs=None, agent_log=None):
        days = self.record_agent_arrivals(full_inputs=full_inputs, agent_log=agent_log)
        # add first turn
        agent_turns = self.threads[self.agent_thread].get_agent_turns()
        self.record_buyer_first_turn(full_inputs=full_inputs, agent_log=agent_log,
                                     days=days, agent_turns=agent_turns)
        # record remaining threads
        self.record_agent_thread(turns=agent_turns, agent_log=agent_log,
                                 thread_id=self.agent_thread, full_inputs=full_inputs)

    def generate_agent_log(self, params):
        # if the agent is a buyer
        if not self.agent:
            return None
        agent_log = AgentLog(byr=self.byr)
        full_inputs = params['inputs'][agent_log.model_name]
        if self.byr:
            self.generate_buyer_log(full_inputs=full_inputs, agent_log=agent_log)
        else:
            self.generate_seller_log(full_inputs=full_inputs, agent_log=agent_log)
        return agent_log

    def generate_seller_log(self, full_inputs=None, agent_log=None):
        for thread_id, thread_log in self.threads.items():
            agent_turns = thread_log.get_agent_turns()
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
                # print('Thread id: {}'.format(thread_id))
                thread_logs[thread_id] = self.generate_thread_log(thread_id=thread_id, params=params)
        return thread_logs

    def generate_arrival_logs(self, params):
        arrival_logs = dict()
        if params['x_thread'] is None:
            # print('first censored')
            censored = self.generate_censored_arrival(params=params, thread_id=1)
            arrival_logs[1] = censored
        else:
            # print('first real thread')
            num_arrivals = len(params['x_thread'].index)
            for i in range(1, num_arrivals + 1):
                curr_arrival = self.generate_arrival_log(params=params,
                                                         thread_id=i)
                arrival_logs[i] = curr_arrival

            if not self.check_bin(params=params, thread_id=num_arrivals):
                censored = self.generate_censored_arrival(params=params, thread_id=num_arrivals + 1)
                arrival_logs[num_arrivals + 1] = censored
            # else:
            #    print('has bin')
        return arrival_logs

    @staticmethod
    def get_arrival_inputs(params=None, thread_id=None):
        if thread_id == 1:
            arrival_inputs = None
        else:
            full_arrival_inputs = params['inputs'][INTERARRIVAL_MODEL]
            arrival_inputs = populate_test_model_inputs(
                full_inputs=full_arrival_inputs,
                value=thread_id)
        return arrival_inputs

    def generate_censored_arrival(self, params=None, thread_id=None):
        arrival_inputs = self.get_arrival_inputs(params=params,
                                                 thread_id=thread_id)
        check_time = self.arrival_check_time(params=params, thread_id=thread_id)
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
        arrival_inputs = self.get_arrival_inputs(params=params,
                                                 thread_id=thread_id)
        check_time = self.arrival_check_time(params=params, thread_id=thread_id)
        # print(params['x_thread'].columns)
        time = int(params['x_thread'].loc[thread_id, MONTHS_SINCE_LSTG] * MONTH)
        time += self.lookup[START_TIME]
        hist = params['x_thread'].loc[thread_id, BYR_HIST] / 10

        full_hist_inputs = params['inputs'][BYR_HIST_MODEL]
        # print('value: {}'.format(value))
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
        thread_params['inputs'] = subset_inputs(models=OFFER_MODELS, input_data=params['inputs'],
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

    def get_agent_arrival(self, time=None, thread_id=None):
        arrival_time = self.translator.get_agent_arrival(check_time=time, thread_id=thread_id)
        return arrival_time

    def get_inter_arrival(self, thread_id=None, input_dict=None, time=None):
        if time == self.lookup[START_TIME]:
            assert input_dict is None
        else:
            assert input_dict is not None
        true_id = self.translate_arrival(thread_id=thread_id)
        if self.verbose:
            print('true id: {}'.format(true_id))
            print('actual id: {}'.format(thread_id))
        return self.arrivals[true_id].get_inter_arrival(check_time=time,
                                                        input_dict=input_dict)

    def get_hist(self, thread_id=None, input_dict=None, time=None):
        true_id = self.translate_thread(thread_id=thread_id)
        return self.arrivals[true_id].get_hist(check_time=time,
                                               input_dict=input_dict)

    def get_action(self, agent_tuple=None):
        if not self.agent:
            raise RuntimeError("Querying action in non-agent LstgLog")
        return self.agent_log.get_action(agent_tuple=agent_tuple)

    def verify_done(self):
        if not self.agent:
            raise RuntimeError("Verifying empty action queue for "
                               "non-agent LstgLog")
        self.agent_log.verify_done()

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

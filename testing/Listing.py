from testing.Arrival import Arrival
from testing.Thread import Thread
from testing.util import subset_inputs, populate_inputs
from constants import MAX_DELAY_ARRIVAL, DAY
from featnames import START_TIME, DAYS_SINCE_LSTG, BYR_HIST, CON, AUTO, \
    X_THREAD, X_OFFER, THREAD, LSTG, LOOKUP, OFFER_MODELS, INTERARRIVAL_MODEL, BYR_HIST_MODEL


class Listing:

    def __init__(self, params=None):
        """
        :param params: dict
        """
        self.lstg = params[LSTG]
        self.lookup = params[LOOKUP]
        self.arrivals_first = params['arrivals_first']
        self.start_time = int(round(self.lookup[START_TIME]))
        self.verbose = params['verbose']
        self.arrivals = self.generate_arrivals(params)
        self.threads = self.generate_threads(params)

    @property
    def has_arrivals(self):
        return not self.arrivals[1].censored

    def generate_threads(self, params):
        threads = dict()
        for thread_id, arrival in self.arrivals.items():
            if not arrival.censored:
                threads[thread_id] = self.generate_thread(thread_id=thread_id,
                                                          params=params)
        return threads

    def generate_arrivals(self, params):
        arrivals = dict()
        num_arrivals = 0 if params[X_THREAD] is None else len(params[X_THREAD].index)
        if num_arrivals > 0:
            for thread_id in range(1, num_arrivals + 1):
                arrivals[thread_id] = self.generate_arrival(params=params,
                                                            thread_id=thread_id)

        is_bin = self.check_bin(params=params, thread_id=num_arrivals)
        if self.arrivals_first or num_arrivals == 0 or not is_bin:
            thread_id = num_arrivals + 1
            phantom = self.arrivals_first and is_bin
            arrivals[thread_id] = self.generate_censored_arrival(params=params,
                                                                 thread_id=thread_id,
                                                                 phantom=phantom)
        return arrivals

    @staticmethod
    def get_arrival_inputs(params=None, thread_id=None):
        if thread_id == 1:
            arrival_inputs = None
        else:
            full_arrival_inputs = params['inputs'][INTERARRIVAL_MODEL]
            arrival_inputs = populate_inputs(
                full_inputs=full_arrival_inputs,
                value=thread_id)
        return arrival_inputs

    def generate_censored_arrival(self, params=None, thread_id=None, phantom=False):
        if phantom:
            arrival_inputs = None
        else:
            arrival_inputs = self.get_arrival_inputs(params=params,
                                                     thread_id=thread_id)
        check_time = self.arrival_check_time(params=params, thread_id=thread_id)
        time = self.start_time + MAX_DELAY_ARRIVAL
        return Arrival(check_time=check_time,
                       arrival_inputs=arrival_inputs,
                       time=time,
                       first_arrival=thread_id == 1,
                       phantom=phantom)

    def arrival_check_time(self, params=None, thread_id=None):
        if thread_id == 1:
            check_time = self.start_time
        else:
            check_time = int(params[X_THREAD].loc[thread_id - 1, DAYS_SINCE_LSTG] * DAY)
            check_time += self.start_time
        return check_time

    def generate_arrival(self, params=None, thread_id=None):
        arrival_inputs = self.get_arrival_inputs(params=params,
                                                 thread_id=thread_id)
        check_time = self.arrival_check_time(params=params, thread_id=thread_id)
        time = int(params[X_THREAD].loc[thread_id, DAYS_SINCE_LSTG] * DAY)
        time += self.start_time
        hist = params[X_THREAD].loc[thread_id, BYR_HIST]
        full_hist_inputs = params['inputs'][BYR_HIST_MODEL]
        hist_inputs = populate_inputs(full_inputs=full_hist_inputs,
                                      value=thread_id)
        return Arrival(hist=hist,
                       time=time,
                       arrival_inputs=arrival_inputs,
                       hist_inputs=hist_inputs,
                       check_time=check_time,
                       first_arrival=thread_id == 1)

    def generate_thread(self, thread_id=None, params=None):
        thread_params = dict()
        thread_params[X_OFFER] = params[X_OFFER].xs(thread_id,
                                                    level=THREAD,
                                                    drop_level=True)
        thread_params['inputs'] = subset_inputs(models=OFFER_MODELS,
                                                input_data=params['inputs'],
                                                value=thread_id,
                                                level=THREAD)
        return Thread(params=thread_params,
                      arrival_time=self.arrivals[thread_id].time,
                      agent=self.is_agent_thread(thread_id),
                      agent_buyer=self.is_agent_buyer)

    def get_con(self, thread_id=None, turn=None, input_dict=None, time=None):
        """
        :return: np.float
        """
        return self.threads[thread_id].get_con(turn=turn,
                                               time=time,
                                               input_dict=input_dict)

    def get_msg(self, thread_id=None, turn=None, input_dict=None, time=None):
        """
        :return: np.float
        """
        msg = self.threads[thread_id].get_msg(turn=turn,
                                              time=time,
                                              input_dict=input_dict)
        return float(msg)

    def get_delay(self, thread_id=None, turn=None, input_dict=None, time=None):
        """
        :return: int
        """
        return self.threads[thread_id].get_delay(turn=turn,
                                                 time=time,
                                                 input_dict=input_dict)

    def get_inter_arrival(self, thread_id=None, input_dict=None, time=None):
        if time == self.start_time:
            assert input_dict is None
        else:
            assert input_dict is not None
        return self.arrivals[thread_id].get_inter_arrival(check_time=time,
                                                          input_dict=input_dict)

    def get_hist(self, thread_id=None, input_dict=None, time=None):
        return self.arrivals[thread_id].get_hist(check_time=time,
                                                 input_dict=input_dict)

    @staticmethod
    def check_bin(params=None, thread_id=None):
        if thread_id == 0:
            return False
        thread1 = params[X_OFFER].xs(thread_id, level=THREAD)
        if len(thread1.index) == 1:
            return thread1.loc[1, CON] == 1
        elif len(thread1.index) == 2:
            return thread1.loc[2, AUTO] and (thread1.loc[2, CON] == 1)

    def is_agent_thread(self, thread_id=None):
        return False

    @property
    def is_agent_buyer(self):
        return False

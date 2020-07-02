from agent.util import get_agent_name
from constants import DAY
from featnames import LSTG, START_TIME


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
                        return self.thread_l - 1  # query_twice
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
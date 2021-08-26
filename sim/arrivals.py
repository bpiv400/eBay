import numpy as np
from env.Sources import Sources
from env.util import get_clock_feats, sample_categorical
from utils import get_days_since_lstg, load_model
from constants import MAX_DELAY_ARRIVAL, DAY, INTERVAL
from featnames import INTERARRIVAL_MODEL, BYR_HIST_MODEL, DAYS_SINCE_LAST, \
    THREAD_COUNT, DAYS_SINCE_LSTG, CLOCK


class ArrivalInterface:
    def __init__(self):
        self.interarrival_model = load_model(INTERARRIVAL_MODEL)
        self.hist_model = load_model(BYR_HIST_MODEL)

    @staticmethod
    def first_arrival(seconds=None):
        return seconds

    def inter_arrival(self, input_dict=None):
        logits = self.interarrival_model(input_dict).squeeze()
        sample = sample_categorical(logits=logits)
        seconds = int((sample + np.random.uniform()) * INTERVAL)
        return seconds

    def hist(self, input_dict=None):
        params = self.hist_model(input_dict).squeeze()
        return self._sample_hist(params=params)

    @staticmethod
    def _sample_hist(params=None):
        # draw a random uniform for mass at 0
        pi = 1 / (1 + np.exp(-params[0]))  # sigmoid
        if np.random.uniform() < pi:
            hist = 0
        else:
            # draw p of negative binomial from beta
            a = np.exp(params[1])
            b = np.exp(params[2])
            p = np.random.beta(a, b)

            # r for negative binomial, of at least 1
            r = np.exp(params[3]) + 1

            # draw from negative binomial
            hist = np.random.negative_binomial(r, p)
        return hist


class ArrivalQueryStrategy:
    def __init__(self, arrival=None):
        self.arrival = arrival

    def get_first_arrival(self, **kwargs):
        return self.arrival.first_arrival(seconds=kwargs['seconds'])

    def get_inter_arrival(self, **kwargs):
        return self.arrival.inter_arrival(input_dict=kwargs['input_dict'])

    def get_hist(self, **kwargs):
        return self.arrival.hist(input_dict=kwargs['input_dict'])


class ArrivalSources(Sources):
    def __init__(self, x_lstg=None, start_time=None):
        super().__init__(x_lstg=x_lstg)
        self.start_time = start_time
        self.priority = None

    def reset(self):
        self.source_dict[DAYS_SINCE_LAST] = 0.0
        self.source_dict[DAYS_SINCE_LSTG] = 0.0
        self.source_dict[THREAD_COUNT] = 0.0
        self.priority = None

    def update_arrival(self, arrival_time=None, last_arrival_time=None, thread_count=None):
        self.source_dict[THREAD_COUNT] = thread_count
        self.source_dict[CLOCK] = get_clock_feats(arrival_time)
        self.source_dict[DAYS_SINCE_LSTG] = \
            get_days_since_lstg(lstg_start=self.start_time, time=arrival_time)
        self.source_dict[DAYS_SINCE_LAST] = \
            (arrival_time - last_arrival_time) / DAY
        self.priority = arrival_time


class ArrivalSimulator:
    def __init__(self, composer=None, query_strategy=None):
        self.composer = composer
        self.query_strategy = query_strategy

        # to be set later
        self.start_time = None
        self.end_time = None
        self.first_arrival = None
        self.first_arrivals = None
        self.sources = None
        self.thread_count = None
        self.last_arrival_time = None
        self.sim_num = None

    def set_lstg(self, x_lstg=None, first_arrivals=None, start_time=None):
        self.sources = ArrivalSources(x_lstg=x_lstg, start_time=start_time)
        self.first_arrivals = first_arrivals
        self.start_time = start_time
        self.end_time = self.start_time + MAX_DELAY_ARRIVAL
        self.sim_num = -1

    def reset(self):
        self.sources.reset()
        self.thread_count = 0
        self.last_arrival_time = self.start_time
        self.sim_num += 1
        if self.first_arrivals is not None:
            self.first_arrival = self.first_arrivals[self.sim_num]

    def simulate_arrivals(self, arrivals=None):
        self.reset()
        if arrivals is None:
            arrivals = []
            seconds = self.query_strategy.get_first_arrival(
                time=self.start_time,
                seconds=self.first_arrival
            )
            priority = self.start_time + seconds
        else:
            self.thread_count = len(arrivals)
            if self.thread_count > 1:
                self.last_arrival_time = arrivals[-2][0]
            self._update_arrival(priority=arrivals[-1][0])
            priority = self._get_next_arrival()

        while priority < self.end_time:
            self.thread_count += 1
            self._update_arrival(priority)
            hist = self._get_hist()
            arrivals.append((priority, hist))
            priority = self._get_next_arrival()
        return arrivals

    def _get_next_arrival(self):
        input_dict = self.composer.build_input_dict(model_name=INTERARRIVAL_MODEL,
                                                    sources=self.sources())
        seconds = self.query_strategy.get_inter_arrival(
            time=self.last_arrival_time,
            thread_id=self.thread_count + 1,
            input_dict=input_dict
        )
        priority = min(self.sources.priority + seconds, self.end_time)
        return priority

    def _get_hist(self):
        input_dict = self.composer.build_input_dict(model_name=BYR_HIST_MODEL,
                                                    sources=self.sources())
        hist = self.query_strategy.get_hist(
            input_dict=input_dict,
            time=self.sources.priority,
            thread_id=self.thread_count
        )
        return hist

    def _update_arrival(self, priority=None):
        self.sources.update_arrival(arrival_time=priority,
                                    thread_count=self.thread_count,
                                    last_arrival_time=self.last_arrival_time)
        self.last_arrival_time = priority

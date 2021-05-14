import numpy as np
from rlenv.Sources import Sources
from rlenv.util import sample_categorical, get_clock_feats
from utils import load_model, get_days_since_lstg
from rlenv.const import CLOCK_MAP
from constants import INTERVAL_ARRIVAL, MAX_DELAY_ARRIVAL, DAY
from featnames import FIRST_ARRIVAL_MODEL, INTERARRIVAL_MODEL, DAYS_SINCE_LAST, THREAD_COUNT, DAYS_SINCE_LSTG


class ArrivalSources(Sources):
    def __init__(self, x_lstg=None):
        super().__init__(x_lstg=x_lstg)
        self.source_dict[DAYS_SINCE_LAST] = 0.0
        self.source_dict[THREAD_COUNT] = 0.0

    def update_arrival(self, arrival_time=None, last_arrival_time=None,
                       thread_count=None, start_time=None):
        self.source_dict[THREAD_COUNT] = thread_count
        self.source_dict[CLOCK_MAP] = get_clock_feats(arrival_time)
        self.source_dict[DAYS_SINCE_LSTG] = \
            get_days_since_lstg(lstg_start=start_time, time=arrival_time)
        self.source_dict[DAYS_SINCE_LAST] = \
            (arrival_time - last_arrival_time) / DAY


class ArrivalInterface:
    def __init__(self):
        self.firstarrival_model = load_model(FIRST_ARRIVAL_MODEL)
        self.interarrival_model = load_model(INTERARRIVAL_MODEL)

    def first_arrival(self, input_dict=None):
        logits = self.firstarrival_model(input_dict).cpu().squeeze()
        sample = sample_categorical(logits=logits)
        return sample

    def inter_arrival(self, input_dict=None):
        logits = self.interarrival_model(input_dict).cpu().squeeze()
        sample = sample_categorical(logits=logits)
        return sample


class ArrivalQueryStrategy:
    def __init__(self, arrival=None):
        self.arrival = arrival

    def get_first_arrival(self, **kwargs):
        interval = self.arrival.first_arrival(input_dict=kwargs['input_dict'])
        return self._seconds_from_interval(interval)

    def get_inter_arrival(self, **kwargs):
        interval = self.arrival.inter_arrival(input_dict=kwargs['input_dict'])
        return self._seconds_from_interval(interval)

    @staticmethod
    def _seconds_from_interval(interval=None):
        return int((interval + np.random.uniform()) * INTERVAL_ARRIVAL)


class ArrivalSimulator:
    def __init__(self, start_time=None, x_lstg=None, composer=None, query_strategy=None):
        self.start_time = start_time
        self.end_time = self.start_time + MAX_DELAY_ARRIVAL
        self.x_lstg = x_lstg
        self.composer = composer
        self.query_strategy = query_strategy

    def simulate_arrivals(self):
        arrivals = []
        thread_count = 0
        last_arrival_time = self.start_time
        sources = ArrivalSources(x_lstg=self.x_lstg)

        input_dict = self.composer.build_input_dict(model_name=FIRST_ARRIVAL_MODEL,
                                                    sources=sources(),
                                                    turn=None)
        seconds = self.get_first_arrival(time=self.start_time,
                                         input_dict=input_dict)
        priority = self.start_time + seconds

        while priority < self.end_time:
            thread_count += 1
            arrivals.append(priority)

            sources.update_arrival(start_time=self.start_time,
                                   arrival_time=priority,
                                   thread_count=thread_count,
                                   last_arrival_time=last_arrival_time)
            last_arrival_time = priority
            input_dict = self.composer.build_input_dict(model_name=INTERARRIVAL_MODEL,
                                                        sources=sources(),
                                                        turn=None)
            seconds = self.get_inter_arrival(time=last_arrival_time,
                                             thread_id=thread_count + 1,
                                             input_dict=input_dict)
            priority = min(priority + seconds, self.end_time)

        return arrivals

    def get_first_arrival(self, *args, **kwargs):
        return self.query_strategy.get_first_arrival(*args, **kwargs)

    def get_inter_arrival(self, *args, **kwargs):
        return self.query_strategy.get_inter_arrival(*args, **kwargs)

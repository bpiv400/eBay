from rlenv.Sources import Sources
from rlenv.util import get_clock_feats
from utils import get_days_since_lstg
from rlenv.const import CLOCK_MAP
from constants import MAX_DELAY_ARRIVAL, DAY
from featnames import INTERARRIVAL_MODEL, BYR_HIST_MODEL, DAYS_SINCE_LAST, \
    THREAD_COUNT, DAYS_SINCE_LSTG


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
        self.source_dict[CLOCK_MAP] = get_clock_feats(arrival_time)
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
        self.logits0 = None
        self.sources = None
        self.thread_count = None
        self.last_arrival_time = None

    def set_lstg(self, x_lstg=None, logits0=None, start_time=None):
        self.sources = ArrivalSources(x_lstg=x_lstg, start_time=start_time)
        self.logits0 = logits0
        self.start_time = start_time
        self.end_time = self.start_time + MAX_DELAY_ARRIVAL

    def reset(self):
        self.sources.reset()
        self.thread_count = 0
        self.last_arrival_time = self.start_time

    def simulate_arrivals(self, arrivals=None):
        self.reset()
        if arrivals is None:
            arrivals = []
            seconds = self.get_first_arrival(time=self.start_time, logits=self.logits0)
            priority = self.start_time + seconds
        else:
            self.thread_count = len(arrivals)
            if self.thread_count > 1:
                self.last_arrival_time = arrivals[-2]
            self._update_arrival(priority=arrivals[-1])
            priority = self._get_next_arrival()

        while priority < self.end_time:
            self.thread_count += 1
            self._update_arrival(priority)
            hist = self._get_hist()
            arrivals.append((priority, hist))
            priority = self._get_next_arrival()
        return arrivals

    def get_first_arrival(self, *args, **kwargs):
        return self.query_strategy.get_first_arrival(*args, **kwargs)

    def get_inter_arrival(self, *args, **kwargs):
        return self.query_strategy.get_inter_arrival(*args, **kwargs)

    def get_hist(self, *args, **kwargs):
        return self.query_strategy.get_hist(*args, **kwargs)

    def _get_next_arrival(self):
        input_dict = self.composer.build_input_dict(model_name=INTERARRIVAL_MODEL,
                                                    sources=self.sources())
        seconds = self.get_inter_arrival(time=self.last_arrival_time,
                                         thread_id=self.thread_count + 1,
                                         input_dict=input_dict)
        priority = min(self.sources.priority + seconds, self.end_time)
        return priority

    def _get_hist(self):
        input_dict = self.composer.build_input_dict(model_name=BYR_HIST_MODEL,
                                                    sources=self.sources())
        hist = self.get_hist(input_dict=input_dict,
                             time=self.sources.priority,
                             thread_id=self.thread_count)
        return hist

    def _update_arrival(self, priority=None):
        self.sources.update_arrival(arrival_time=priority,
                                    thread_count=self.thread_count,
                                    last_arrival_time=self.last_arrival_time)
        self.last_arrival_time = priority

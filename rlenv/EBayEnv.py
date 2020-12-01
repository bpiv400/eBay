from collections import namedtuple
from constants import DAY, MAX_DELAY_ARRIVAL
from featnames import DEC_PRICE, ACC_PRICE, START_PRICE, DELAY, START_TIME, BYR, \
    INTERARRIVAL_MODEL, BYR_HIST_MODEL
from utils import get_days_since_lstg
from rlenv.Heap import Heap
from rlenv.time.TimeFeatures import TimeFeatures
from rlenv.time.Offer import Offer
from rlenv.events.Event import Event
from rlenv.events.Arrival import Arrival
from rlenv.generate.Recorder import Recorder
from rlenv.Sources import ArrivalSources
from rlenv.Sources import ThreadSources
from rlenv.events.Thread import Thread
from rlenv.const import INTERACT, ACC_IND, CON, MSG, REJ_IND, OFF_IND, \
    ARRIVAL, FIRST_OFFER, OFFER_EVENT, DELAY_EVENT
from rlenv.util import get_clock_feats, get_con_outcomes, need_msg, model_str
from agent.AgentLoader import AgentLoader


class EBayEnv:
    Outcome = namedtuple('outcome', ['sale', 'price', 'days', 'thread'])

    def __init__(self, **kwargs):
        # for printing
        self.verbose = kwargs['verbose']

        # features
        self.x_lstg = None
        self.lookup = None
        self.p_arrival = None

        # lstg params
        self.end_time = None
        self.start_time = None
        self.thread_counter = 1
        self.outcome = None
        self.last_arrival_time = None

        # classes
        self.time_feats = TimeFeatures()
        self.queue = Heap(entry_type=Event)
        self.composer = kwargs['composer']
        self.query_strategy = kwargs['query_strategy']
        self.recorder = None if 'recorder' not in kwargs else kwargs['recorder']
        if 'loader' in kwargs:
            self.loader = kwargs['loader']
        else:
            x_lstg_cols = self.composer.x_lstg_cols
            self.loader = AgentLoader(x_lstg_cols=x_lstg_cols, byr=False)

    def reset(self, push_arrival=True):
        self.last_arrival_time = self.start_time
        self.queue.reset()
        self.time_feats.reset()
        self.outcome = None
        self.thread_counter = 1
        if push_arrival:
            event = Arrival(priority=self.start_time,
                            sources=ArrivalSources(x_lstg=self.x_lstg))
            self.queue.push(event)
        if self.verbose:
            Recorder.print_lstg(self.lookup)

    def has_next_lstg(self):
        return self.loader.has_next()

    def next_lstg(self):
        """
        Sample a new lstg from the file and set lookup and x_lstg series
        """
        x_lstg, lookup, p_arrival = self.loader.next_lstg()
        self.x_lstg = self.composer.decompose_x_lstg(x_lstg)
        self.lookup = lookup
        self.start_time = int(self.lookup[START_TIME])
        self.end_time = self.start_time + MAX_DELAY_ARRIVAL
        self.query_strategy.update_p_arrival(p_arrival=p_arrival)
        if self.recorder is not None:
            # update listing in recorder
            self.recorder.update_lstg(lookup=self.lookup,
                                      lstg=self.loader.lstg)
        return self.loader.lstg

    def run(self):
        while True:
            event = self.queue.pop()
            if self.verbose:
                Recorder.print_next_event(event)
            if INTERACT and event.type != ARRIVAL:
                input('Press Enter to continue...\n')
            if self.is_agent_turn(event):
                return event, False
            else:
                lstg_complete = self.process_event(event)
                if lstg_complete:
                    return event, True

    def process_event(self, event):
        if event.type == ARRIVAL:
            # print('processing arrival')
            return self.process_arrival(event)
        elif event.type == FIRST_OFFER:
            # print('processing first offer for thread {} for turn 1'.format(event.thread_id))
            return self.process_first_offer(event)
        elif event.type == OFFER_EVENT:
            return self.process_offer(event)
        elif event.type == DELAY_EVENT:
            # print('processing delay for thread {} for turn {}'.format(event.thread_id,event.turn))
            return self.process_delay(event)
        else:
            raise NotImplementedError()

    def record(self, event, byr_hist=None):
        if self.recorder is None:
            if self.verbose and byr_hist is None:
                Recorder.print_offer(event)
        else:
            if byr_hist is None:
                time_feats = self.time_feats.get_feats(thread_id=event.thread_id,
                                                       time=event.priority)
                self.recorder.add_offer(event=event, time_feats=time_feats)
            else:
                self.recorder.start_thread(thread_id=event.thread_id,
                                           byr_hist=byr_hist,
                                           time=event.priority)

    def process_offer(self, event):
        # check whether the lstg expired, censoring this offer
        if self.is_lstg_expired(event):
            return self.process_lstg_expiration(event)
        # otherwise check whether this offer corresponds to an expiration rej
        slr_offer = event.turn % 2 == 0
        if event.thread_expired():
            if slr_offer:
                self.process_slr_expire(event)
            else:
                self.process_byr_expire(event)
            return False
        # otherwise updates thread features
        self.prepare_offer(event)
        # generate concession and msg if necessary
        offer = self.get_offer_outcomes(event, slr=slr_offer)
        # print(str(offer))
        return self.process_post_offer(event, offer)

    def process_post_offer(self, event, offer):
        slr_offer = event.turn % 2 == 0
        # print('summary right before record')
        # print(event.summary())
        self.record(event)
        # check whether the offer is an acceptance
        if event.is_sale():
            self._process_sale(offer)
            return True
        # otherwise check whether the offer is a rejection
        elif event.is_rej():
            if slr_offer:
                self._process_slr_rej(event, offer)
            else:
                self._process_byr_rej(offer)
            return False
        else:
            if not slr_offer:
                auto = self._check_slr_autos(offer.price)
                if auto == ACC_IND:
                    self._process_slr_auto_acc(event)
                    return True
                elif auto == REJ_IND:
                    self._process_slr_auto_rej(event, offer)
                    return False
            self.time_feats.update_features(offer=offer)
            self._init_delay(event)
            return False

    def _process_sale(self, offer):
        if offer.player == BYR:
            start_norm = offer.price
        else:
            start_norm = 1 - offer.price
        sale_price = start_norm * self.lookup[START_PRICE]
        days = (offer.time - self.start_time) / DAY
        self.outcome = self.Outcome(True, sale_price, days, offer.thread_id)
        self.empty_queue()

    def process_first_offer(self, event):
        """
        Processes the buyer's first offer in a thread
        :return:
        """
        # assign thread id
        event.set_id(self.thread_counter)
        self.thread_counter += 1

        # prepare sources and features
        sources = ThreadSources(x_lstg=self.x_lstg)
        days_since_lstg = get_days_since_lstg(lstg_start=self.start_time,
                                              time=event.priority)
        time_feats = self.time_feats.get_feats(time=event.priority,
                                               thread_id=event.thread_id)
        sources.prepare_hist(time_feats=time_feats,
                             clock_feats=get_clock_feats(event.priority),
                             days_since_lstg=days_since_lstg)
        # sample history
        input_dict = self.composer.build_input_dict(model_name=BYR_HIST_MODEL,
                                                    sources=sources(),
                                                    turn=None)
        hist = self.get_hist(input_dict=input_dict,
                             time=event.priority,
                             thread_id=event.thread_id)
        # print
        if self.verbose:
            print('Thread {} initiated | Buyer hist: {}'.format(event.thread_id, hist))

        # update features with history
        event.init_thread(sources=sources, hist=hist)
        self.record(event, byr_hist=hist)
        return self.process_offer(event)

    def prepare_offer(self, event):
        # if offer not expired and thread still active, prepare this turn's inputs
        if event.turn != 1:
            time_feats = self.time_feats.get_feats(thread_id=event.thread_id,
                                                   time=event.priority)
            event.init_offer(time_feats=time_feats)
        return False

    def process_arrival(self, event):
        """
        Updates queue with results of an Arrival Event
        :param event: Event corresponding to current event
        :return: boolean indicating whether the lstg has ended
        """
        if self.is_lstg_expired(event):
            return self.process_lstg_expiration(event)

        # update sources with clock feats
        event.update_arrival(thread_count=self.thread_counter - 1,
                             last_arrival_time=self.last_arrival_time)

        # call model to sample inter arrival time and update arrival check priority
        if event.priority == self.start_time:
            seconds = self.get_first_arrival(time=event.priority,
                                             thread_id=self.thread_counter)
        else:
            seconds = self.get_inter_arrival(event=event)
        check_in_time = event.priority
        event.priority = min(event.priority + seconds, self.end_time)

        if self.verbose:
            print('Next arrival at {}'.format(event.priority))

        # if a buyer arrives, create a thread at the arrival time
        if event.priority < self.end_time:
            self.last_arrival_time = check_in_time
            self.queue.push(Thread(priority=event.priority))
        self.queue.push(event)
        return False

    def process_byr_expire(self, event):
        event.byr_expire()
        self.record(event)
        offer_params = {
            'thread_id': event.thread_id,
            'time': event.priority,
            'player': BYR
        }
        self.time_feats.update_features(offer=Offer(params=offer_params, rej=True))

    def process_slr_expire(self, event):
        # update sources with new clock and features
        time_feats = self.time_feats.get_feats(thread_id=event.thread_id,
                                               time=event.priority)
        event.init_offer(time_feats=time_feats)
        offer = event.slr_expire_rej()
        self.record(event)
        self.time_feats.update_features(offer=offer)
        self._init_delay(event)
        return False

    def process_delay(self, event):
        # no need to check expiration since this must occur at the same time as the previous offer
        input_dict = self.get_delay_input_dict(event)
        delay_seconds = self.get_delay(input_dict=input_dict,
                                       turn=event.turn,
                                       thread_id=event.thread_id,
                                       time=event.priority)
        # Test environment returns None when delay model is mistakenly called
        if delay_seconds is None:
            # print("No delay returned; exiting listing.")
            return True
        event.update_delay(seconds=delay_seconds)
        self.queue.push(event)
        return False

    def is_agent_turn(self, event):
        raise NotImplementedError()

    def is_lstg_expired(self, event):
        return event.priority >= self.end_time

    def process_lstg_expiration(self, event):
        """
        Checks whether the lstg has expired by the time of the event
        If so, record the reward as negative insertion fees
        :param event: rlenv.Event subclass
        :return: boolean
        """
        self.outcome = self.Outcome(False, 0, MAX_DELAY_ARRIVAL, None)
        self.queue.push(event)
        self.empty_queue()
        if self.verbose:
            print('Lstg expired')
        return True

    def empty_queue(self):
        while not self.queue.empty:
            self.queue.pop()

    def _check_slr_autos(self, norm):
        if norm >= self.lookup[ACC_PRICE] / self.lookup[START_PRICE]:
            return ACC_IND
        elif norm < self.lookup[DEC_PRICE] / self.lookup[START_PRICE]:
            return REJ_IND
        else:
            return OFF_IND

    def _process_byr_rej(self, offer):
        self.time_feats.update_features(offer=offer)

    def _process_slr_rej(self, event, offer):
        self.time_feats.update_features(offer=offer)
        self._init_delay(event)

    def _process_slr_auto_rej(self, event, offer):
        self.time_feats.update_features(offer=offer)
        event.change_turn()
        time_feats = self.time_feats.get_feats(thread_id=event.thread_id,
                                               time=event.priority)
        offer = event.slr_auto_rej(time_feats=time_feats)
        self.record(event)
        self.time_feats.update_features(offer=offer)
        self._init_delay(event)

    def _process_slr_auto_acc(self, event):
        offer = event.slr_auto_acc()
        self.record(event)
        self._process_sale(offer)

    def _init_delay(self, event):
        event.change_turn()
        event.init_delay(self.start_time)
        self.queue.push(event)

    def get_con(self, *args, **kwargs):
        return self.query_strategy.get_con(*args, **kwargs)

    def get_msg(self, *args, **kwargs):
        return self.query_strategy.get_msg(*args, **kwargs)

    def get_first_arrival(self, *args, **kwargs):
        return self.query_strategy.get_first_arrival(*args, **kwargs)

    def get_inter_arrival(self, event=None):
        input_dict = self.composer.build_input_dict(model_name=INTERARRIVAL_MODEL,
                                                    sources=event.sources(),
                                                    turn=None)
        return self.query_strategy.get_inter_arrival(time=event.priority,
                                                     thread_id=self.thread_counter,
                                                     input_dict=input_dict)

    def get_hist(self, *args, **kwargs):
        return self.query_strategy.get_hist(*args, **kwargs)

    def get_delay(self, *args, **kwargs):
        return self.query_strategy.get_delay(*args, **kwargs)

    def get_offer_outcomes(self, event, slr=False):
        # sample concession
        model_name = model_str(CON, turn=event.turn)
        input_dict = self.composer.build_input_dict(model_name=model_name,
                                                    sources=event.sources(),
                                                    turn=event.turn)
        con = self.get_con(input_dict=input_dict,
                           time=event.priority,
                           turn=event.turn,
                           thread_id=event.thread_id)
        con_outcomes = get_con_outcomes(con=con,
                                        sources=event.sources(),
                                        turn=event.turn)
        # update features
        offer = event.update_con_outcomes(con_outcomes=con_outcomes)
        # sample msg if necessary
        if need_msg(con, slr):
            model_name = model_str(MSG, turn=event.turn)
            input_dict = self.composer.build_input_dict(model_name=model_name,
                                                        sources=event.sources(),
                                                        turn=event.turn)
            msg = self.get_msg(input_dict=input_dict,
                               time=event.priority,
                               turn=event.turn,
                               thread_id=event.thread_id)
            event.update_msg(msg=msg)
        return offer

    def get_delay_input_dict(self, event=None):
        model_name = model_str(DELAY, turn=event.turn)
        input_dict = self.composer.build_input_dict(model_name=model_name,
                                                    sources=event.sources(),
                                                    turn=event.turn)
        return input_dict

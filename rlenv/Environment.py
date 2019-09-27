"""
Environment for that simulates eBay market

Ultimately, the goal is to implement a parallel version of this environment
that executes queues for multiple sellers in parallel
"""
import random
import math
import h5py
import pandas as pd
import numpy as np
import torch
import utils
from Arrival import Arrival
from FirstOffer import FirstOffer
from ThreadEvent import ThreadEvent
from EventQueue import EventQueue
from TimeFeatures import TimeFeatures
import time_triggers
from env_consts import *
from event_types import *
from env_utils import *
from constants import INTERVAL_COUNTS, INTERVAL, MAX_DELAY
from SimulatorInterface import SimulatorInterface
import rlenv.model_names as model_names


# TODO: Redo documentation when all is said and done
# TODO: Ensure default value tensors are calculated and stored correctly and efficiently


class Environment:
    """
    Class implementing eBay bargaining reinforcement learning environment
    Currently, environment doesn't integrate learner

    Input Files:
        experiments.csv: spreadsheet containing parameters of the rl experiment
        (id,arrive_model,byr_model,slr_model,byr_us,byr_hist,interval) -- not all well-defined yet
        input.h5: h5py file containing a dataset called 'slrs' 1-d np array containing all slr ids
        and a 2-dimensional np.array of lstg features for each slr in datasets named after the slrs
    Attributes:
        experiment_id: unique experiment id to extract features from experiments.csv
        time_feats: TimeFeatures Object
        slr_queues: dictionary mapping slr id to EventQueue object containing all events
        for that slr
        input_file: input file
        slrs: 1-d np.array containing all slr ids
        params: dictionary containing parameters extracted from experiments.csv
        thread_counter: integer giving a count of the number of open threads, ensures each
        thread has a unique id
        interface: SimulatorInterface object
        sales: dictionary mapping lstg id to sale
    Public Functions:
        iterate_queue: executes a defined number of steps in some slr's queue
        initialize_slr: initializes the queue of lstgs for a given slr
    Private functions:
        _load_params: loads experiment parameters from experiments.csv
        _get_event_data: constructs 'data' argument for some given next event
        _make_offer: constructs the 'offer' input to TimeFeatures update functions
        _process_delay: executes a delay event
        _process_offer: executes an offer event
        _process_arrival: executes an arrival / newItem events
    """
    def __init__(self, experiment_id):
        """

        :param experiment_id: integer giving experiment id
        """
        # experiment level data
        self.experiment_id = experiment_id
        self.input_file = h5py.File(LSTG_FILENAME, 'r')
        self.k = self.input_file['lstg'].size[0]
        self.params = self._load_params()
        self.thread_counter = 0
        self.lstgs = []
        self.lstg_slice = None

        # lstg level data
        self.time_feats = None
        self.end_time = None  # can just be an integer now
        self.queue = None
        self.curr_lstg = None
        self.sales = dict()

        # interface data
        self.interface = SimulatorInterface(params=self.params)

    def _load_params(self):
        """
        Loads dictionary of parameters associated with the current experiment
        from experiments spreadsheet

        :return: dictionary containing parameter values
        """
        params = pd.read_csv(EXPERIMENT_PATH)
        params.set_index('id', drop=True, inplace=True)
        params = params.loc[self.experiment_id, :].to_dict()
        return params

    def _initialize_lstg(self, index):
        """
        Sets up a lstg for the environment after the initial creation of the environment
        or after the lstg sells

        :param index: index of the lstg in the lstg slice object
        :return: NA
        """
        self.consts = self.lstg_slice[index, :]
        start_time = self.consts[LSTG_COLS['start_days']]
        self.queue = EventQueue()
        self.queue.push(Arrival(start_time))
        self.time_feats = TimeFeatures()
        self.end_time = start_time + MONTH * self.params[RELIST_COUNT]

    def prepare_simulation(self, lstgs):
        """
        Initializes a priority queue for each lstg in lstgs
        Function should be called by master node when environment is first
        created

        :param lstgs: list of lstg indices in h5df file
        :return: NA
        """
        # type cast to list to ensure lstg_slice is 2 dimensional
        if type(lstgs) == int:
            lstgs = [lstgs]
        self.lstgs = lstgs
        if max(lstgs) >= self.k:
            raise RuntimeError("lstg {} invalid".format(max(lstgs)))
        self.lstg_slice = self.input_file[self.lstgs, :]
        self.input_file.close()
        self.lstg_slice = torch.from_numpy(self.lstg_slice)
        for lstg in lstgs:
            self.sales[lstg] = []

    def _simulate_market(self):
        """
        Runs a simulation of the ebay market for all lstgs in self.lstgs until
        sale or expiration after 12 listings (1 listing and 11 automatic re-listings)

        :return: Accumulates sale price and sale time in self.rewards
        """
        for i, lstg in enumerate(self.lstgs):
            self._initialize_lstg(i)
            self._simulate_lstg()

    def _simulate_lstg(self):
        """
        Runs a simulation of a single lstg until sale or expiration
        :return: NA
        """
        complete = False
        while not complete:
            complete = self._process_event(self.queue.pop())

    def simulate(self):
        """
        Runs self.params[SIM_COUNT] simulations of the ebay market and accumuates
        rewards and times sold for each

        :param lstgs:
        :return:
        """
        for _ in range(self.params[SIM_COUNT]):
            self._simulate_market()

    def _process_event(self, event):
        if event.type == ARRIVAL:
            return self._process_arrival(event)
        elif event.type == FIRST_OFFER:
            return self._process_first_offer(event)
        elif event.type == BUYER_OFFER:
            return self._process_offer(event, byr=True)
        elif event.type == SELLER_OFFER:
            return self._process_offer(event, byr=False)
        elif event.type == BUYER_DELAY:
            return self._process_delay(event, byr=True)
        elif event.type == SELLER_DELAY:
            return self._process_delay(event, byr=False)

    def _process_first_offer(self, event):
        """
        Processes the buyer's first offer in a thread

        :param event:
        :return:
        """
        # expiration
        if self._lstg_expiration(event):
            return True
        # check buy it now
        if event.bin:
            self._process_sale(norm=1, time=event.priority)
            return True
        sources = self._make_offer_sources(event)
        cn, norm, split, hidden_cn = self.interface.cn(sources=sources, hidden=None, slr=False)
        sources[OUTCOMES_MAP][[1, 2, 3]] = [cn, norm, split]

        model_name = model_str(model_names.MSG, byr=True)
        msg, hidden_msg = self.interface.offer_indicator(model_name, sources=sources,
                                                         hidden=None)
        sources[OUTCOMES_MAP][6] = msg
        model_name = model_str(model_names.RND, byr=True)
        rnd, hidden_rnd = self.interface.offer_indicator(model_name, sources=sources,
                                                         hidden=None)
        sources[OUTCOMES_MAP][4] = rnd
        model_name = model_str(model_names.NINE, byr=True)
        nine, hidden_nine = self.interface.offer_indicator(model_name, sources=sources,
                                                           hidden=None, sample=(rnd == 0))
        sources[OUTCOMES_MAP][5] = nine

        if norm < self.consts[LSTG_COLS['accept']]:
            hidden = self._make_hidden(hidden_msg=hidden_msg, hidden_cn=hidden_cn, hidden_nine=hidden_nine,
                                       hidden_rnd=hidden_rnd)
            self.time_feats.update_features(time_trigger=time_triggers.OFFER, thread_id=event.thread_id,
                                            offer={
                                                'time': event.priority,
                                                'type': model_names.BYR_PREFIX,
                                                'price': norm
                                            })
            if event.sources[O_OUTCOMES_MAP][2] <= self.consts[LSTG_COLS['decline']]:
                return self._prepare_delay(event, sources=sources, hidden=hidden, byr=True)
            else:
                self._increment_sources(sources)
                return self._process_slr_rej_early(event, sources=sources, hidden=hidden, auto=True, exp=False)
        else:
            return self._process_sale(norm=norm, time=event.priority)

    @staticmethod
    def _make_hidden(hidden_msg=None, hidden_cn=None, hidden_nine=None, hidden_rnd=None):
        """

        :param hidden_msg:
        :param hidden_cn:
        :param hidden_nine:
        :param hidden_rnd:
        :return:
        """
        hidden = dict()
        # byr models
        hidden[model_str(model_names.MSG, byr=True)] = hidden_msg
        hidden[model_str(model_names.CON, byr=True)] = hidden_cn
        hidden[model_str(model_names.NINE, byr=True)] = hidden_nine
        hidden[model_str(model_names.RND, byr=True)] = hidden_rnd
        hidden[model_str(model_names.ACC, byr=True)] = None
        hidden[model_str(model_names.REJ, byr=True)] = None
        # slr models
        hidden[model_str(model_names.MSG, byr=False)] = None
        hidden[model_str(model_names.CON, byr=False)] = None
        hidden[model_str(model_names.NINE, byr=False)] = None
        hidden[model_str(model_names.RND, byr=False)] = None
        hidden[model_str(model_names.ACC, byr=False)] = None
        hidden[model_str(model_names.REJ, byr=False)] = None
        return hidden

    def _process_slr_rej_early(self, event, sources=None, hidden=None, auto=False, exp=False):
        """
        Process slr automatic rejection or seller expiration rejection

        :param event:
        :param sources:
        :param hidden:
        :return:
        """
        if auto:
            sources[DIFFS_MAP] = torch.zeros(len(TIME_FEATS))
            sources[OUTCOMES_MAP] = AUTO_REJ_OUTCOMES
        elif exp:
            sources[OUTCOMES_MAP] = EXP_REJ_OUTCOMES
            sources[CLOCK_MAP] = utils.get_clock_feats(event.priority,
                                                       start_days=self.consts[LSTG_COLS['start_days']],
                                                       arrival=False, delay=False)
            sources[TIME_MAP] = self.time_feats.get_feats(thread_id=event.thread_id, time=event.priority)
            sources[DIFFS_MAP] = sources[TIME_MAP] - sources[O_TIME_MAP]
        # indicate there's a rejection and its automatic
        sources[OUTCOMES_MAP][1:6] = sources[L_OUTCOMES_MAP][1:6]
        for model_name in [model_str(model_names.ACC, byr=False), model_str(model_names.REJ, byr=False)]:
            _, hdn = self.interface.offer_indicator(model_name, sources=sources,
                                                    hidden=hidden[model_name], sample=False)
            hidden[model_name] = hdn
        return self._process_slr_rej(event, sources=sources, hidden=hidden, con_rej=False)

    def _process_slr_rej(self, event, sources=None, hidden=None, con_rej=False):
        """
        Processes an ordinary seller rejection (see _process_slr_rej_early for handling automatic
        and expiration rejections)

        :param event:
        :param sources:
        :param hidden:
        :param con_rej:
        :return:
        """
        for base_model in model_names.OFFER_NO_PREFIXES:
            if base_model in [model_names.ACC, model_names.REJ]:
                continue
            if con_rej and base_model == model_names.CON:
                continue
            model_name = model_str(base_model, byr=False)
            if base_model != model_names.CON:
                _, hdn = self.interface.offer_indicator(model_name, sources=sources,
                                                        hidden=hidden[model_name],
                                                        sample=False)
            else:
                _, _, _, hdn = self.interface.cn(sources=sources, hidden=None,
                                                 slr=False, sample=False)
            hidden[model_name] = hdn
        self.time_feats.update_features(time_trigger=time_triggers.SLR_REJECTION, thread_id=event.thread_id,
                                        offer={
                                            'time': event.priority,
                                            'type': model_names.SLR_PREFIX,
                                            'price': sources[OUTCOMES_MAP][2]
                                        })
        return self._prepare_delay(event, sources=sources, hidden=hidden, byr=True)

    @staticmethod
    def _increment_sources(sources, turn=1):
        """
        Updates sources dictionary for use in the next turn. This consists of
        pushing current turn maps to _other maps and pushing _other maps to
        _last maps

        :param sources: dict
        :return: None
        """
        # push other sources
        sources[L_TIME_MAP] = sources[O_TIME_MAP]
        sources[L_CLOCK_MAP] = sources[O_CLOCK_MAP]
        sources[L_OUTCOMES_MAP] = sources[O_OUTCOMES_MAP]
        sources[O_DIFFS_MAP] = sources[DIFFS_MAP]
        # push current sources
        sources[O_TIME_MAP] = sources[TIME_MAP]
        sources[O_OUTCOMES_MAP] = sources[OUTCOMES_MAP]
        sources[O_CLOCK_MAP] = sources[CLOCK_MAP]
        # turn indicators
        if turn % 2 == 0:
            num_turns = 2
        else:
            num_turns = 3
        sources[TURN_IND_MAP] = torch.zeros(num_turns).float()
        if turn < 5:
            ind = math.floor(turn / 2)
            sources[TURN_IND_MAP][ind] = 1

    def _prepare_delay(self, event, sources=None, hidden=None, byr=False):
        """
        Adds the next offer to the queue in some thread
        :param sources:
        :param hidden:
        :param byr: boolean indicating whether the next offer is a byr offer
        :return: False, indicating the thread isn't over
        """
        self._increment_sources(sources, turn=event.turn)
        turn = event.turn + 1
        if byr:
            event_type = BUYER_DELAY
        else:
            event_type = SELLER_DELAY
        next_event = ThreadEvent(event.priority, sources=sources, hidden=hidden, turn=turn,
                                 thread_id=event.thread_id, event_type=event_type)
        sources[PERIODS_MAP] = torch.zeros(1).float()
        sources[model_str(DELAY, byr=byr)] = None
        self.queue.push(next_event)
        return False

    def _process_offer(self, event, byr=False):
        """

        :param event:
        :return:
        """
        if self._lstg_expiration(event):
            return True
        hidden, sources = event.hidden, event.sources
        model_name = model_str(model_names.ACC, byr=byr)
        # acceptance
        acc, hdn = self.interface.offer_indicator(model_name, sources=sources,
                                                  hidden=hidden[model_name], sample=True)
        if acc == 1:
            if byr:
                norm = 1 - sources[O_OUTCOMES_MAP][2]
            else:
                norm = sources[O_OUTCOMES_MAP][2]
            return self._process_sale(norm=norm, time=event.priority)
        hidden[model_name] = hdn
        # rejection
        model_name = model_str(model_names.REJ, byr=byr)
        rej, hdn = self.interface.offer_indicator(model_name, sources=sources,
                                                  hidden=hidden[model_name], sample=True)
        hidden[model_name] = hdn
        if rej == 1:
            if byr:
                self.time_feats.update_features(trigger_type=time_triggers.BYR_REJECTION,
                                                thread_id=event.thread_id)
            else:
                return self._process_slr_rej(event, sources=sources, hidden=hidden)
        # concession




    @staticmethod
    def _is_turn_expired(periods, turn=1, byr=False):
        """

        :param event:
        :param byr:
        :return:
        """
        if not byr:
            max_periods = INTERVAL_COUNTS[model_names.SLR_PREFIX]
        elif turn == 7:
            max_periods = INTERVAL_COUNTS['{}_7'.format(model_names.BYR_PREFIX)]
        else:
            max_periods = INTERVAL_COUNTS[model_names.BYR_PREFIX]
        return periods >= max_periods

    def _thread_expiration(self, event, byr=False):
        """
        Checks whether the thread has expired due to an offer timing out

        If a buyer has allowed the turn to timeout,
        :param event:
        :return:
        """
        if Environment._is_turn_expired(event.sources[PERIODS_MAP], turn=event.turn, byr=byr):
            if byr:
                self.time_feats.update_features(time_trigger=time_triggers.BYR_REJECTION,
                                                thread_id=event.thread_id,
                                                offer={
                                                    'time': event.priority,
                                                    'type': model_names.BYR_PREFIX,
                                                    'price': 0
                                                })
            else:
                self.time_feats.update_features(time_trigger=time_triggers.SLR_REJECTION,
                                                thread_id=event.thread_id,
                                                offer={
                                                    'time': event.priority,
                                                    'type': model_names.SLR_PREFIX,
                                                    'price': event.sources[L_OUTCOMES_MAP][2]
                                                })
            return True
        else:
            return False

    def _process_delay(self, event, byr=False):
        """
        Handles buyer and seller delay events (i.e. a buyer or seller waiting to decide
        whether to respond to some open offer)

        :param event:
        :return:
        """
        if self._lstg_expiration(event):
            return True
        elif self._thread_expiration(event, byr=byr):
            return False

        sources = event.sources
        hidden = event.hidden
        sources[TIME_MAP] = self.time_feats.get_feats(thread_id=event.thread_id, time=event.priority)
        sources[CLOCK_MAP] = utils.get_clock_feats(event.priority, self.consts[LSTG_COLS['start_days']],
                                                   arrival=False, delay=True)
        sources[DIFFS_MAP] = sources[TIME_MAP] - sources[O_TIME_MAP]
        model_name = model_str(model_names.DELAY, byr=byr)
        delay, hdn_del = self.interface.offer_indicator(model_name, sources=sources,
                                                        hidden=hidden[model_name], sample=True)
        hidden[model_name] = hdn_del
        if delay == 0:
            if byr:
                turn_name = model_names.BYR_PREFIX
                next_type = BUYER_DELAY
            else:
                turn_name = model_names.SLR_PREFIX
                next_type = SELLER_DELAY
            priority = event.priority + INTERVAL[turn_name]
            sources[PERIODS_MAP] += 1
            next_event = ThreadEvent(event_type=next_type, priority=priority, sources=sources, hidden=hidden,
                                     turn=event.turn)
        else:
            if byr:
                turn_name = model_names.BYR_PREFIX
                next_type = BUYER_OFFER
            else:
                turn_name = model_names.SLR_PREFIX
                next_type = SELLER_OFFER
            last_interval = np.random.randint(0, INTERVAL[turn_name], size=1)
            priority = event.priority + last_interval
            hidden[model_name] = None
            sources[OUTCOMES_MAP][0] = Environment._get_delay(sources[PERIODS_MAP], last_interval,
                                                              turn=event.turn)
            next_event = ThreadEvent(event_type=next_type, priority=priority, sources=sources, hidden=hidden,
                                     turn=event.turn)
        self.queue.push(next_event)
        return False

    @staticmethod
    def _get_delay(periods, last_interval, turn):
        """

        :param periods:
        :param last_interval:
        :param turn:
        :return:
        """
        if turn == 7:
            max_periods = INTERVAL_COUNTS['{}_7'.format(model_names.BYR_PREFIX)]
            period_len = INTERVAL[model_names.BYR_PREFIX]
        elif turn % 2 == 0:
            max_periods = INTERVAL_COUNTS[model_names.SLR_PREFIX]
            period_len = INTERVAL[model_names.SLR_PREFIX]
        else:
            max_periods = INTERVAL_COUNTS[model_names.BYR_PREFIX]
            period_len = INTERVAL[model_names.BYR_PREFIX]
        period_count = periods + last_interval / period_len
        return period_count / max_periods

    def _process_arrival(self, event):
        """
        Updates queue with results of an Arrival Event

        :param event: Event corresponding to current event
        :return: boolean indicating whether the lstg has ended
        """
        if self._lstg_expiration(event):
            return True

        sources = self._make_base_sources(event, arrival=True, delay=False)
        num_byrs, hidden_days = self.interface.days(sources=sources,
                                                    hidden=event.hidden_days)

        if num_byrs > 0:
            loc = self.interface.loc(sources=sources, num_byrs=num_byrs)
            hist = self.interface.hist(sources=sources, byr_us=loc)
            self._add_attr_sources(sources, byr_hist=hist, byr_us=loc)
            sec = self.interface.sec(sources=sources, num_byrs=num_byrs)
            bin = self.interface.bin(sources=sources, num_byrs=num_byrs)
            # place each into the queue
            for i in range(num_byrs):
                byr_attr = torch.zeros(2).float()
                byr_attr[0] = loc[i]
                byr_attr[1] = hist[i]
                priority = event.priority + int(sec[i] * DAY)
                self.thread_counter += 1
                offer_event = FirstOffer(priority, byr_attr=byr_attr,
                                         thread_id=self.thread_counter, bin=bin[i])
                self.queue.push(offer_event)
        # Add arrival check
        priority = event.priority + DAY
        arrive = Arrival(priority=priority, hidden_days=hidden_days)
        self.queue.push(arrive)
        return False

    def _lstg_expiration(self, event):
        """
        Checks whether the lstg has expired by the time of the event
        If so, record the reward as negative insertion fees
        :param event: rlenv.Event subclass
        :return: boolean
        """
        if event.priority >= self.end_time:
            profit = -1 * self._insertion_fees(event.priority)
            self.sales[self.curr_lstg].append(False, profit, event.priority)
            return True
        else:
            return False

    def _insertion_fees(self, time):
        """
        Returns the insertion fees the seller paid to lst an item up to the
        time of event.priority
        :param time: int giving the time of the sale
        :return: Float
        """
        dur = time - self.consts[LSTG_COLS['start_days']] * DAY
        periods = math.ceil(dur / MONTH)
        periods = min(periods, self.params[RELIST_COUNT])
        fees = periods * ANCHOR_STORE_INSERT
        return fees

    def _make_base_sources(self, time, arrival=True, delay=False):
        """
        Constructs the sources dictonary for the days model

        :param time: time of the arrival event
        :return: dict containing LST_MAP, TIME_MAP, CLOCK_MAP
        """
        sources = dict()
        start_days = self.consts[LSTG_COLS['start_days']]
        sources[CLOCK_MAP] = utils.get_clock_feats(time, start_days, arrival=True,
                                                   delay=False)
        sources[LSTG_MAP] = self.consts
        sources[TIME_MAP] = self.time_feats.get_feats(time=time)
        return sources

    def _make_offer_sources(self, event):
        """
        Creates a source dictionary for the first offer in a thread
        :param event: rlenv.Event
        :return: dict
        """
        sources = self._make_base_sources(event.priority, arrival=False, delay=False)
        sources[BYR_ATTR_MAP] = event.byr_attr
        sources[OUTCOMES_MAP] = torch.zeros(len(BYR_OUTCOMES)).float()
        sources[TURN_IND_MAP] = torch.zeros(3).float()
        sources[TURN_IND_MAP][0] = 1
        sources[DIFFS_MAP] = torch.zeros(len(TIME_FEATS)).float()
        # other turn maps
        sources[O_OUTCOMES_MAP] = ZERO_SLR_OUTCOMES
        sources[O_OUTCOMES_MAP][[4, 5]] = self.consts[[LSTG_COLS['start_round'],
                                                       LSTG_COLS['start_nines']]].float()
        sources[O_TIME_MAP] = torch.zeros(len(TIME_FEATS)).float()
        sources[O_DIFFS_MAP] = torch.zeros(len(TIME_FEATS)).float()
        sources[O_CLOCK_MAP] = torch.zeros(len(OFFER_CLOCK_FEATS)).float()
        sources[O_CLOCK_MAP][1:len(DAYS_CLOCK_FEATS)] = self.consts[START_CLOCK_MAP]
        # last turn maps
        sources[L_OUTCOMES_MAP] = torch.zeros(len(BYR_OUTCOMES)).float()
        sources[L_TIME_MAP] = torch.zeros(len(TIME_FEATS)).float()
        return sources

    @staticmethod
    def _add_attr_sources(sources, byr_us=None, byr_hist=None):
        """
        Adds byr_us and byr_hist maps to the sources dictionary
        for some set of arrival models

        :param sources: dictionary containing tensors that will be used to construct
        input
        :param byr_us: integer indicator giving whether the byr is from the us
        :param byr_hist: integer giving the number of best offer threads the buyer
        has participated in in the past
        :return: None (modifies dictionary in place)
        """
        sources[BYR_US_MAP] = byr_us
        sources[BYR_HIST_MAP] = byr_hist

    def _process_sale(self, norm=0, sale_price=None, time=0):
        """
        Adds a sale with the given norm (offer price / start price)
        and time to the sales tracking dictionary

        :param norm: float giving offer price / start price
        :param time: int giving time of sale
        :return: None
        """
        if sale_price is None:
            sale_price = norm * self.consts[LSTG_COLS['start']]
        insertion_fees = self._insertion_fees(time)
        value_fee = self._value_fee(sale_price, time)
        net = sale_price - insertion_fees - value_fee
        self.sales[self.curr_lstg].append(True, net, time)
        return True

    def _value_fee(self, price, time):
        """
        Computes the value fee. For now, just set to 10%
        of sale price, pending refinement decisions

        # TODO: Implement full meta conditional logic
        :param price: price of sale
        :param time: int giving time of sale
        :return: float
        """
        return .1 * price




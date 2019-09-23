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
from event_types import (BUYER_OFFER, SELLER_OFFER, ARRIVAL, BUYER_DELAY, SELLER_DELAY)
from EventQueue import EventQueue
from TimeFeatures import TimeFeatures
import time_triggers
from env_consts import *
from SimulatorInterface import SimulatorInterface


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

    def _process_offer(self, event, byr=False):
        """

        :param event:
        :return:
        """
        pass

    def _process_delay(self, event, byr=False):
        """

        :param event:
        :return:
        """
        pass

    def _process_arrival(self, event):
        """
        Updates queue with results of an Arrival Event

        :param event: Event corresponding to current event
        :return: boolean indicating whether the lstg has ended
        """
        if self._lstg_expiration(event):
            return True

        sources = self._make_days_sources(event)
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
        :param event:
        :return:
        """
        dur = time - self.consts[LSTG_COLS['start_days']] * DAY
        periods = math.ceil(dur / MONTH)
        periods = min(periods, self.params[RELIST_COUNT])
        fees = periods * ANCHOR_STORE_INSERT
        return fees

    def _make_days_sources(self, time):
        """
        Constructs the sources dictonary for the days model

        :param time: time of the arrival event
        :return:
        """
        sources = dict()
        start_days = self.consts[LSTG_COLS['start_days']]
        sources[CLOCK_MAP] = utils.get_clock_feats(time, start_days, arrival=True,
                                                   delay=False)
        sources[LSTG_MAP] = self.consts
        sources[TIME_MAP] = self.time_feats.get_feats(time=time)
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

    def _process_sale(self, norm=0, time=0):
        """
        Adds a sale with the given norm (offer price / start price)
        and time to the sales tracking dictionary

        :param norm: float giving offer price / start price
        :param time: int giving time of sale
        :return: None
        """
        sale_price = norm * self.consts[LSTG_COLS['start']]
        insertion_fees = self._insertion_fees(time)
        value_fee = self._value_fee(sale_price, time)
        net = sale_price - insertion_fees - value_fee
        self.sales[self.curr_lstg].append(True, net, time)

    def _value_fee(self, price, time):
        """
        Computes the value fee. For now, just set to 10%
        of sale price, pending refinement decisions

        :param price: price of sale
        :param time: int giving time of sale
        :return: float
        """
        return .1 * price




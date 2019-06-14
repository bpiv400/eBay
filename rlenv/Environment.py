"""
Environment for that simulates eBay market

Ultimately, the goal is to implement a parallel version of this environment
that executes queues for multiple sellers in parallel
"""
import random
import h5py
import pandas as pd
import numpy as np
from NewItem import NewItem
from Arrival import Arrival
from ThreadEvent import ThreadEvent
from env_constants import LSTG_FEATS_MAP, CONSTS_MAP, CONSTS
from event_types import NEW_ITEM, BUYER_OFFER, SELLER_OFFER, ARRIVAL, BUYER_DELAY, SELLER_DELAY
from EventQueue import EventQueue
from TimeFeatures import TimeFeatures
import time_triggers
import env_utils
from SimulatorInterface import SimulatorInterface

INPUT_PATH = 'data/rl/input.h5'
EXPERIMENT_PATH = 'repo/rlenv/experiments.csv'
EXPIRATION = 15
BYR_DELAY_TIME = 14 * 24 * 60 * 60
SLR_DELAY_TIME = 2 * 24 * 60 * 60

# TODO: Fix lstg end time definition for unsold lstgs after some end_time feature is added to chunk output
# TODO: Write tests
# TODO: Integrate RL


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
        self.experiment_id = experiment_id
        self.slr_queues = dict()
        self.time_feats = TimeFeatures(expiration=EXPIRATION)
        self.input_file = h5py.File(INPUT_PATH, 'r')
        self.slrs = self.input_file['slrs']
        self.params = self._load_params()
        self.thread_counter = 0
        self.interface = SimulatorInterface(params=self.params)
        self.sales = dict()

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

    def initialize_slr(self, slr):
        """
        Initializes a priority queue for a given slr with all listings

        :param slr: integer giving slr id--must be unique for all sellers
        :return: NA
        """
        event_list = []
        data = self.input_file[str(slr)]
        assert(len(data.shape) == 2)
        const_cols = [LSTG_FEATS_MAP(col) for col in CONSTS]
        for i in range(data.shape[0]):
            lstg_data = data[i, :]
            ids = env_utils.get_ids(slr, lstg_data)
            event_list.append(NewItem(lstg_data[LSTG_FEATS_MAP['start_date']], ids,
                                      end_time=lstg_data[LSTG_FEATS_MAP['end_time']],
                                      consts=lstg_data[const_cols]))
        queue = EventQueue(interval=self.params['interval'], init=event_list)
        self.slr_queues[slr] = queue

    @staticmethod
    def _get_event_data(next_type=None, event=None, hidden=None, offer=None, delay=None):
        """
        Generates data object for the next event given the previous event. See
        Arrival.py, and ThreadEvent.py for details of what each
        event type expects in the data object

        :param event: Event object containing the previous event
        :param next_type: one of the event types from event_types.py giving the next event
        :param hidden: hidden state dictionary after processing the previous event. If the
        previous event is an offer, hidden is a dictionary containing the hidden states relevant
        to that offer. If the next type is an arrival, hidden is a representation of the arrival
        process hidden state(s)
        :param offer: If previous event was an offer, dictionary containing
         details of the offer made as a result of processing the previous event.
         If the previous event was an Arrival or NewOffer, np.array with size 2
         where elements are byr_us, byr_hist
        :param delay: integer giving the number of seconds that the next event occurs
        after the previous one
        :return: dictionary
        """
        prev_type = event.type
        data = dict()
        data['lstg_expiration'] = event.lstg_expiration
        if next_type == ARRIVAL:
            data['consts'] = event.consts
            data['hidden'] = hidden
        elif (prev_type == NEW_ITEM or prev_type == ARRIVAL) and next_type == BUYER_OFFER:
            data['consts'] = np.append(event.consts, offer)
            data['hidden'] = {
                'slr':
                    {
                        'delay': None,
                        'concession': None,
                        'round': None
                    },
                'byr':
                    {
                        'delay': None,
                        'concession': None,
                        'round': None
                    }
            }
            data['prev_slr_offer'] = None
            data['prev_byr_offer'] = None
            data['prev_byr_delay'] = 0
            data['prev_slr_delay'] = 0
            data['delay'] = 0
        elif prev_type == BUYER_OFFER and next_type == SELLER_DELAY:
            data['delay'] = 0
            data['hidden'] = {
                'slr': event.hidden['slr'].copy(),
                'byr': hidden
            }
            data['prev_byr_delay'] = event.delay
            data['prev_slr_delay'] = event.prev_slr_delay
            data['prev_byr_offer'] = offer
            data['prev_slr_offer'] = event.prev_slr_offer
        elif prev_type == SELLER_OFFER and next_type == BUYER_DELAY:
            data['delay'] = 0
            data['hidden'] = {
                'slr': hidden,
                'byr': event.hidden['byr'].copy()
            }
            data['prev_slr_delay'] = event.delay
            data['prev_byr_delay'] = event.prev_byr_delay
            data['prev_byr_offer'] = event.prev_byr_offer
            data['prev_slr_offer'] = offer
        elif (prev_type == SELLER_DELAY and next_type == SELLER_DELAY) or \
                (prev_type == BUYER_DELAY and next_type == BUYER_DELAY):
            data['hidden'] = {
                'byr': event.hidden['byr'].copy(),
                'slr': event.hidden['slr'].copy()
            }
            if prev_type == SELLER_DELAY:
                data['hidden']['slr']['delay'] = hidden
            else:
                data['hidden']['byr']['delay'] = hidden
            data['prev_byr_delay'] = event.prev_byr_delay
            data['prev_slr_delay'] = event.prev_slr_delay
            data['prev_byr_offer'] = event.prev_byr_offer
            data['prev_slr_offer'] = event.prev_slr_offer
            data['delay'] = event.delay + delay
        elif (prev_type == SELLER_DELAY and next_type == SELLER_OFFER) \
                or (prev_type == BUYER_DELAY and next_type == BUYER_OFFER):
            data['hidden'] = {
                'byr': event.hidden['byr'].copy(),
                'slr': event.hidden['slr'].copy()
            }
            if prev_type == SELLER_DELAY:
                data['hidden']['slr']['delay'] = None
            else:
                data['hidden']['byr']['delay'] = None
            data['delay'] = event.delay + delay
            data['prev_byr_delay'] = event.prev_byr_delay
            data['prev_slr_delay'] = event.prev_slr_delay
            data['prev_byr_offer'] = event.prev_byr_offer
            data['prev_slr_offer'] = event.prev_slr_offer
        return data

    def iterate_queue(self, slr_id, steps):
        """
        Executes a given number of events in one of the slr_queues
        :param slr_id: int id for target slr
        :param steps: int giving number of steps
        :return:
        """
        queue = self.slr_queues[slr_id]
        for _ in range(steps):
            event = queue.pop()
            event_type = event.type
            if event_type == NEW_ITEM or event_type == ARRIVAL:
                self._process_arrival(event, queue=queue)
            elif event_type == BUYER_OFFER or event_type == SELLER_OFFER:
                self._process_offer(event, queue=queue)
            elif event_type == BUYER_DELAY or SELLER_DELAY:
                self._process_delay(event, queue=queue)
            else:
                raise NotImplementedError('Invalid Event Type')

    def _lstg_expired(self, event):
        """
        Checks whether the lstg has expired prior to the current event
        If it has but it remains in the time feature object, dispatch
        expiration event

        :param event: Event object
        :return: boolean denoting whether the lstg has expired
        """
        if not self.time_feats.lstg_active(event.ids):
            return True
        elif event.priority > event.lstg_expiration:
            self.time_feats.update_features(trigger_type=time_triggers.LSTG_EXPIRATION,
                                            ids=event.ids,
                                            offer=None)
            return True
        else:
            return False

    @staticmethod
    def _make_offer(offer, byr=False, time=None):
        """
        Makes a dictionary with characteristics of an offer for use in time_features update

        :param offer: output of SimulatorInterface.byr(slr)_offer
        :param byr: boolean denoting whether this is a buyer offer
        :param time: integer denoting the time of the offer object
        :return: dictionary
        """
        time_offer = {
            'byr' : byr,
            'price': offer['price'],
            'time': time
        }
        return time_offer

    def _process_offer(self, event, queue=None):
        """
        Process an offer (buyer or seller), update time valued features, close listing/update rewards as necessary

        :param event: Event corresponding to current event
        :param queue: EventQueue instance
        :return: NA
        """
        if self._lstg_expired(event):
            pass
        if event.type == BUYER_OFFER:
            if event.delay > BYR_DELAY_TIME:
                self.time_feats.update_features(trigger_type=time_triggers.THREAD_EXPIRATION, ids=event.ids)
            else:
                time_feats = self.time_feats.get_feats(event.ids, event.priority)
                offer, hidden = self.interface.buyer_offer(consts=None, hidden=event.hidden['byr'],
                                                           prev_slr_offer=event.prev_slr_offer,
                                                           time_feats=time_feats,
                                                           prev_byr_offer=event.prev_byr_offer,
                                                           prev_slr_delay=event.prev_slr_delay,
                                                           prev_byr_delay=event.prev_byr_delay,
                                                           delay=event.delay)
                time_offer = Environment._make_offer(offer, byr=True)
                # handle buyer rejection
                if offer['concession'] == 0:
                    self.time_feats.update_features(trigger_type=time_triggers.BYR_REJECTION,
                                                    ids=event.ids,
                                                    offer=None)
                # handle acceptance
                elif offer['concession'] == 1:
                    self.time_feats.update_features(trigger_type=time_triggers.SALE,
                                                    ids=event.ids, offer=time_offer)
                    self.sales[event.ids['lstg']] = offer['price']
                else:
                    self.time_feats.update_features(trigger_type=time_triggers.BUYER_OFFER,
                                                    ids=event.ids, offer=time_offer)
                    # check whether the offer should be auto-accepted/rejected
                    if offer['price'] >= event.consts[CONSTS_MAP['accept_price']]:
                        self.time_feats.update_features(trigger_type=time_triggers.SALE, ids=event.ids,
                                                        offer=time_offer)
                        self.sales[event.ids['lstg']] = offer['price']
                        pass
                    elif offer['price'] <= event.consts[CONSTS_MAP['decline_price']]:
                        self.time_feats.update_features(trigger_type=time_triggers.SLR_REJECTION, ids=event.ids)
                        # TODO figure out representation of auto-rejection offer, update _get_event_data as necessary
                        data = self._get_event_data(next_type=BUYER_DELAY, event=event, hidden=hidden, offer=offer)
                        delay_type = BUYER_DELAY
                    else:
                        data = self._get_event_data(next_type=SELLER_DELAY, event=event, hidden=hidden, offer=offer)
                        delay_type = SELLER_DELAY

                    delay = ThreadEvent(priority=event.priority, ids=event.ids.copy(),
                                        data=data, event_type=SELLER_DELAY)
                    queue.push(delay)
        else:
            if event.delay > SLR_DELAY_TIME:
                self.time_feats.update_features(trigger_type=time_triggers.THREAD_EXPIRATION, ids=event.ids)
            else:
                time_feats = self.time_feats.get_feats(event.ids, event.priority)
                offer, hidden = self.interface.slr_offer(consts=None, hidden=event.hidden['slr'],
                                                         time_feats=time_feats,
                                                         prev_slr_offer=event.prev_slr_offer,
                                                         prev_byr_offer=event.prev_byr_offer,
                                                         prev_slr_delay=event.prev_slr_delay,
                                                         prev_byr_delay=event.prev_byr_delay,
                                                         delay=event.delay)
                time_offer = Environment._make_offer(offer, byr=True)
                # handle acceptance
                if offer['concession'] == 1:
                    self.time_feats.update_features(trigger_type=time_triggers.SALE, ids=event.ids, offer=time_offer)
                    self.sales[event.ids['lstg']] = offer['price']
                else:
                    if offer['concession'] == 0:
                        # handle seller rejection
                        self.time_feats.update_features(trigger_type=time_triggers.SLR_REJECTION,
                                                        ids=event.ids,
                                                        offer=None)
                    else:
                        self.time_feats.update_features(trigger_type=time_triggers.SELLER_OFFER,
                                                        ids=event.ids, offer=time_offer)
                    data = self._get_event_data(next_type=BUYER_DELAY, event=event, hidden=hidden, offer=offer)
                    delay = ThreadEvent(priority=event.priority, ids=event.ids.copy(), data=data,
                                        event_type=BUYER_DELAY)
                    queue.push(delay)

    def _process_delay(self, event, queue=None):
        """
        Processes a buyer or seller delay event

        :param event:
        :param queue:
        :return:
        """
        if not self._lstg_expired(event.ids):
            pass
        time_feats = self.time_feats.get_feats(event.ids, event.priority)
        if event.type == BUYER_DELAY:
            if event.delay > BYR_DELAY_TIME:
                self.time_feats.update_features(trigger_type=time_triggers.THREAD_EXPIRATION, ids=event.ids,
                                                offer=None)
                pass
            else:
                realization, hidden = self.interface.buyer_delay(consts=event.consts,
                                                                 time_feats=time_feats,
                                                                 delay=event.delay,
                                                                 prev_byr_offer=event.prev_byr_offer,
                                                                 prev_slr_offer=event.prev_slr_offer,
                                                                 hidden=event.hidden['byr']['delay'])
                if realization == 1:
                    next_type = BUYER_OFFER
                    delay = random.randint(1, self.params['interval'])
                else:
                    next_type = BUYER_DELAY
                    delay = self.params['interval']
        else:
            if event.delay > SLR_DELAY_TIME:
                self.time_feats.update_features(trigger_type=time_triggers.THREAD_EXPIRATION, ids=event.ids,
                                                offer=None)
                pass
            else:
                realization, hidden = self.interface.seller_delay(consts=event.consts,
                                                                  time_feats=time_feats,
                                                                  delay=event.delay,
                                                                  prev_byr_offer=event.prev_byr_offer,
                                                                  prev_slr_offer=event.prev_slr_offer,
                                                                  hidden=event.hidden['slr']['delay'])
                if realization == 1:
                    next_type = SELLER_OFFER
                    delay = random.randint(1, self.params['interval'])
                else:
                    next_type = SELLER_DELAY
                    delay = self.params['interval']
        data = self._get_event_data(next_type=next_type, event=event, hidden=hidden, offer=None, delay=delay)
        next_event = ThreadEvent(priority=event.priority + delay,
                                 ids=event.ids, data=data, event_type=next_type)
        queue.push(next_event)

    def _process_arrival(self, event, queue=None):
        """
        Updates queue with results of a NewItem or Arrival Event

        :param event: Event corresponding to current event
        :param queue: EventQueue instance
        :return: NA
        """
        # simulate arrival time, insert buyer offer
        if event.type == NEW_ITEM:
            self.time_feats.initialize_time_feats(ids=event.ids,
                                                  start_price=event.consts[CONSTS_MAP['start_price']])
        time_feats = self.time_feats.get_feats(event.ids, event.priority)
        realization, hidden = self.interface.arrival_time(consts=event.consts,
                                                          time_feats=time_feats,
                                                          hidden=event.hidden)
        if realization == 1:
            # ADD BUYER OFFER
            ids = event.ids.copy()
            ids['thread_id'] = self.thread_counter
            self.thread_counter += 1
            time = random.randint(1, self.params['interval'])
            time += event.priority
            #################
            # HARD CODING byr_us and byr_hist until arrival model integration
            # treat first element as byr_us, second as byr_hist
            offer = np.array(1, 2)
            ##############
            data = self._get_event_data(next_type=BUYER_OFFER, event=event, offer=offer)
            offer = ThreadEvent(ids=ids, data=data, priority=time, event_type=BUYER_OFFER)
            queue.push(offer)
        # ADD ARRIVAL CHECK
        time = event.priority + self.params['interval']
        data = self._get_event_data(next_type=ARRIVAL, event=event, hidden=hidden)
        arrive = Arrival(ids=event.ids, priority=time, data=data)
        queue.push(arrive)


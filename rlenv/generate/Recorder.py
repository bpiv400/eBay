import numpy as np
import pandas as pd
from rlenv.events.Thread import Thread
from rlenv.const import FIRST_OFFER
from constants import MAX_DELAY_ARRIVAL
from featnames import START_TIME, START_PRICE, TIME_FEATS, MSG, CON, \
    LSTG, THREAD, INDEX, BYR_HIST, DEC_PRICE, CLOCK

OFFER_COLS = [LSTG, THREAD, INDEX, CLOCK, CON, MSG] + TIME_FEATS
THREAD_COLS = [LSTG, THREAD, BYR_HIST, CLOCK]
INDEX_COLS = [LSTG, THREAD, INDEX]


class Recorder:
    def __init__(self, verbose):
        self.verbose = verbose

        self.lstg = None
        self.start_time = None
        self.start_price = None
        self.sim = None

    def update_lstg(self, lookup, lstg):
        """
        Resets lstg that the recorder is tracking data about
        :param pd.Series lookup: lstg meta data
        :param int lstg: listing id
        """
        self.lstg = lstg
        self.sim = -1
        self.start_time = lookup[START_TIME]
        self.start_price = lookup[START_PRICE]
        if self.verbose:
            self.print_lstg(lookup)

    def reset_sim(self):
        self.sim += 1
        # print('Simulation {}'.format(self.sim))

    @staticmethod
    def print_offer(event):
        """
        Prints data about the offer if verbose
        """
        con, norm, msg = event.summary()
        if event.turn > 1:
            days, delay = event.delay_outcomes()
        else:
            days, delay = 0, 0
        byr = event.turn % 2 != 0
        if con == 0:
            if delay == 1 and not byr:
                otype = 'expiration rejection'
            elif delay == 0 and not byr:
                otype = 'automatic rejection'
            else:
                otype = 'standard rejection'
        elif con == 100:
            if delay == 0 and not byr:
                otype = 'automatic acceptance'
            else:
                otype = 'standard acceptance'
        else:
            otype = 'standard offer'
        print('Thread: {} | Offer: {} | clock: {}'.format(event.thread_id,
                                                          event.turn, event.priority))
        if event.turn > 1:
            print('Days: {}, Delay: {}'.format(days, delay))
        print('Offer type: {}'.format(otype))
        print('Concession: {} | norm: {} | message: {}'.format(con, norm, msg))

    @staticmethod
    def print_lstg(lookup):
        print("\nLSTG: {}".format(int(lookup[LSTG])))
        print('start time: {} | end time: {}'.format(
            int(lookup[START_TIME]), int(lookup[START_TIME] + MAX_DELAY_ARRIVAL)))
        norm_dec = lookup[DEC_PRICE] / lookup[START_PRICE]
        print('Start price: {} | decline price: {} | norm decline price: {}'.format(
            lookup[START_PRICE], lookup[DEC_PRICE], norm_dec))

    @staticmethod
    def print_next_event(event):
        print('NEXT EVENT DRAWING...')
        print_string = 'Type: {} at {}'.format(event.type, event.priority)
        if isinstance(event, Thread) and event.type != FIRST_OFFER:
            print_string += ' | Thread: {} | Turn: {}'.format(
                event.thread_id, event.turn)
        print(print_string)

    @staticmethod
    def print_agent_turn(con=None, delay=None):
        if delay is not None:
            print('AGENT TURN: con: {} | delay : {}'.format(con, delay))
        else:
            print('AGENT TURN: con: {}'.format(con))

    def print_sale(self, sale, price, dur):
        """
        Print info about the given sale data if self.verbose
        """
        if self.verbose:
            if sale:
                print('Item sold for: {} in {} seconds'.format(price, dur))
            else:
                print('Item did not sell')

    @staticmethod
    def record2frame(record, cols):
        """
        Converts the given list of lists into a dataframe with the given column names
        :param [[numeric]] record:
        :param [str] cols:
        :return: pd.DataFrame with columns given by cols
        """
        return pd.DataFrame(data=record, columns=cols)

    def start_thread(self, thread_id=None, time=None, byr_hist=None):
        raise NotImplementedError()

    def add_offer(self, event, time_feats=None):
        raise NotImplementedError()

    def construct_output(self):
        raise NotImplementedError()

    def reset_recorders(self):
        raise NotImplementedError()


class OutcomeRecorder(Recorder):
    def __init__(self, verbose=None, byr_agent=False):
        super().__init__(verbose)
        self.byr_agent = byr_agent
        self.offers = None
        self.threads = None
        self.reset_recorders()

    def reset_recorders(self):
        # tracker dictionaries
        self.offers = []
        self.threads = []

    def start_thread(self, thread_id=None, byr_hist=None, time=None, agent=False):
        """
        Records an arrival
        :param thread_id: int giving the thread id
        :param byr_hist: float giving byr history decile
        :param int time: time of the offer
        :param bool agent: True if agent byr starts thread
        """
        row = [self.lstg, thread_id, byr_hist, time]
        if self.byr_agent:
            row += [agent]
        self.threads.append(row)

    def add_offer(self, event=None, time_feats=None):
        # change ordering if OFFER_COLS changes
        summary = event.summary()
        con, norm, msg = summary
        if event.turn == 1:
            assert con > 0
        row = [self.lstg,
               event.thread_id,
               event.turn,
               event.priority,
               con,
               msg
               ]
        row += list(time_feats)
        self.offers.append(row)
        if self.verbose:
            self.print_offer(event)

    @staticmethod
    def compress_common_cols(cols, frames, dtypes):
        for i, col in enumerate(cols):
            for frame in frames:
                frame[col] = frame[col].astype(dtypes[i])

    def construct_output(self):
        # convert both lists to dataframes
        self.offers = self.record2frame(self.offers, OFFER_COLS)
        thread_cols = THREAD_COLS
        if self.byr_agent:
            thread_cols += ['byr_agent']
        self.threads = self.record2frame(self.threads, thread_cols)

        # offers dataframe
        self.offers[INDEX] = self.offers[INDEX].astype(np.uint8)
        self.offers[CON] = self.offers[CON].astype(np.uint8)
        self.offers[MSG] = self.offers[MSG].astype(bool)
        for name in TIME_FEATS:
            if 'offers' in name or 'count' in name:
                self.offers[name] = self.offers[name].astype(np.uint8)

        # common columns in offers and threads dataframes
        self.compress_common_cols([LSTG, THREAD, CLOCK],
                                  [self.threads, self.offers],
                                  [np.int32, np.uint16, np.int32])
        self.offers.set_index(INDEX_COLS, inplace=True).sort_index()
        self.threads.set_index(INDEX_COLS[:-1], inplace=True).sort_index()
        assert np.all(self.offers.xs(1, level='index')[CON] > 0)

        return dict(offers=self.offers, threads=self.threads)

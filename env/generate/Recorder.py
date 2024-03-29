import numpy as np
import pandas as pd
from env.events.Thread import Thread
from constants import MAX_DELAY_ARRIVAL
from featnames import START_TIME, START_PRICE, TIME_FEATS, MSG, CON, \
    LSTG, SIM, THREAD, INDEX, BYR_HIST, DEC_PRICE, ACC_PRICE, CLOCK, \
    X_THREAD, X_OFFER, IS_AGENT

OFFER_COLS = [LSTG, SIM, THREAD, INDEX, CLOCK, CON, MSG] + TIME_FEATS
THREAD_COLS = [LSTG, SIM, THREAD, BYR_HIST, CLOCK]
INDEX_COLS = [LSTG, SIM, THREAD, INDEX]

DTYPES = {LSTG: np.int32, SIM: np.uint8, THREAD: np.uint8, INDEX: np.uint8,
          BYR_HIST: np.uint32, MSG: bool, CON: np.uint8, CLOCK: np.int32}
for name in TIME_FEATS:
    if 'offers' in name or 'count' in name:
        DTYPES[name] = np.uint8


class Recorder:
    def __init__(self, verbose):
        self.verbose = verbose

        self.lstg = None
        self.start_time = None
        self.start_price = None
        self.sim = None

    def update_lstg(self, lookup=None, lstg=None, sim=None):
        """
        Resets lstg that the recorder is tracking data about
        :param pd.Series lookup: lstg meta data
        :param int lstg: listing id
        :param int sim: simulation number
        """
        self.lstg = lstg
        self.start_time = lookup[START_TIME]
        self.start_price = lookup[START_PRICE]
        self.sim = sim

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
    def print_lstg(lookup=None, sim=None):
        print("\n\nLSTG: {}, SIM: {}".format(int(lookup[LSTG]), sim))
        print('start time: {} | end time: {}'.format(
            int(lookup[START_TIME]), int(lookup[START_TIME] + MAX_DELAY_ARRIVAL)))
        norm_dec = lookup[DEC_PRICE] / lookup[START_PRICE]
        norm_acc = lookup[ACC_PRICE] / lookup[START_PRICE]
        print('Start price: {} | dec price: {} ({}) | acc price: {} ({})'.format(
            lookup[START_PRICE], lookup[DEC_PRICE], norm_dec,
            lookup[ACC_PRICE], norm_acc))

    @staticmethod
    def print_next_event(event):
        print_string = 'Type: {} at {}'.format(event.type, event.priority)
        if isinstance(event, Thread) and event.turn > 1:
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
        df = pd.DataFrame(data=record, columns=cols)
        for c in df.columns:
            if c in DTYPES:
                df[c] = df[c].astype(DTYPES[c])
        return df

    def start_thread(self, thread_id=None, time=None, byr_hist=None):
        raise NotImplementedError()

    def add_offer(self, event, time_feats=None):
        raise NotImplementedError()

    def construct_output(self):
        raise NotImplementedError()

    def reset_recorders(self):
        raise NotImplementedError()


class OutcomeRecorder(Recorder):
    def __init__(self, verbose=None, byr=False):
        super().__init__(verbose)
        self.byr = byr
        self.offers = None
        self.threads = None
        self.thread_cols = THREAD_COLS
        if byr:
            self.thread_cols += [IS_AGENT]
        self.reset_recorders()

    def reset_recorders(self):
        # tracker dictionaries
        self.offers = []
        self.threads = []

    def start_thread(self, thread_id=None, byr_hist=None, time=None, is_agent=False):
        """
        Records an arrival
        :param thread_id: int giving the thread id
        :param byr_hist: float giving byr history decile
        :param int time: time of the offer
        :param bool is_agent: whether agent is acting as buyer
        """
        row = [self.lstg, self.sim, thread_id, byr_hist, time]
        if self.byr:
            row += [is_agent]
        self.threads.append(row)

    def add_offer(self, event=None, time_feats=None):
        # change ordering if OFFER_COLS changes
        con, norm, msg = event.summary()
        if event.turn == 1:
            assert con > 0
        row = [self.lstg,
               self.sim,
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

    def add_buyer_walk(self, event=None, time_feats=None):
        assert self.byr
        assert event.turn == 1
        row = [self.lstg, self.sim, event.thread_id, 1,
               event.priority, 0, 0]
        row += list(time_feats)
        self.offers.append(row)
        if self.verbose:
            self.print_offer(event)

    def construct_output(self):
        # convert lists to dataframes
        self.offers = self.record2frame(self.offers, OFFER_COLS)
        self.threads = self.record2frame(self.threads, self.thread_cols)

        # set index
        self.offers.set_index(INDEX_COLS, inplace=True)
        self.threads.set_index(INDEX_COLS[:-1], inplace=True)

        # output dictionary
        return {X_OFFER: self.offers.sort_index(),
                X_THREAD: self.threads.sort_index()}

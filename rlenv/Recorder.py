import pandas as pd
from compress_pickle import dump
from constants import MONTH
from featnames import (START_TIME, START_PRICE, TIME_FEATS,
                       MSG, CON, LSTG, BYR_HIST, ACC_PRICE,
                       DEC_PRICE)

# variable names
INDEX = 'index'
VAL = 'value'
THREAD = 'thread'
CLOCK = 'clock'
SE = 'se'
AVG_PRICE = 'avg_price'
NUM_SALES = 'num_sales'
P_SALE = 'p_sale'
CUT = 'cut'
CENSOR = 'censored'

# for discriminator
OFFER_COLS = [LSTG, THREAD, INDEX, CLOCK, CON, MSG, CENSOR] + TIME_FEATS
THREAD_COLS = [LSTG, THREAD, BYR_HIST, CLOCK]
INDEX_COLS = {False: ['lstg', 'thread', 'index'],
              True: ['lstg', 'sim', 'thread', 'index']}

# for values
VAL_COLS = [LSTG, VAL, SE, AVG_PRICE, NUM_SALES, P_SALE, CUT]


class Recorder:
    def __init__(self, records_path, verbose):
        self.records_path = records_path
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

    def reset_sim(self):
        self.sim += 1

    def dump(self):
        self.records2frames()
        self.compress_frames()
        dump(self.construct_output(), self.records_path)
        self.reset_recorders()

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
        print("LSTG: {}".format(int(lookup[LSTG])))
        print('start time: {} | end time: {}'.format(int(lookup[START_TIME]),
                                                     int(lookup[START_TIME] + MONTH)))
        print(('Start price: {} | accept price: {}' +
              ' | decline price: {}').format(lookup[START_PRICE], lookup[ACC_PRICE],
                                             lookup[DEC_PRICE]))
        norm_acc = lookup[START_PRICE] / lookup[ACC_PRICE]
        norm_dec = lookup[START_PRICE] / lookup[DEC_PRICE] if lookup[DEC_PRICE] > 0 else 0.0
        print('Norm accept price: {} | Norm decline price: {}'.format(norm_acc, norm_dec))

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

    def add_offer(self, event, time_feats=None, censored=False):
        raise NotImplementedError()

    def records2frames(self):
        raise NotImplementedError()

    def construct_output(self):
        raise NotImplementedError()

    def compress_frames(self):
        raise NotImplementedError()

    def reset_recorders(self):
        raise NotImplementedError()



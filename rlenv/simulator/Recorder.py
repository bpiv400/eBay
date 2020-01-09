import pandas as pd
from compress_pickle import dump
from featnames import START_TIME, START_PRICE, TIME_FEATS

# variable names
INDEX = 'index'
VAL = 'value'
THREAD = 'thread'
CLOCK = 'clock'
BYR_HIST = 'byr_hist'
NORM = 'norm'
MESSAGE = 'message'
CON = 'con'
DUR = 'duration'
SALE = 'sale'
LSTG = 'lstg'
SE = 'se'
PRICE = 'price'
AVG_PRICE = 'avg_price'
NUM_SALES = 'num_sales'
P_SALE = 'p_sale'
CUT = 'cut'

# for discriminator
OFFER_COLS = [LSTG, THREAD, INDEX, CLOCK, CON, MESSAGE] + TIME_FEATS
THREAD_COLS = [LSTG, THREAD, BYR_HIST, CLOCK]

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
        raise NotImplementedError()

    def print_offer(self, event, summary):
        """
        Prints data about the offer if verbose
        """
        if self.verbose:
            con, norm, msg, split = summary
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
            print('Concession: {} | normalized offer: {} | raw offer : {} | split: {} | message: {}'.format(
                con, norm, (norm * self.start_price / 100), split, msg))
            if event.turn % 2 == 0:
                auto, exp, rej = event.slr_outcomes()
                print('Auto: {} | Exp: {} | Reject: {}'.format(auto, exp, rej))

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

    def add_offer(self, event, time_feats):
        raise NotImplementedError()

    def records2frames(self):
        raise NotImplementedError()

    def construct_output(self):
        raise NotImplementedError()

    def compress_frames(self):
        raise NotImplementedError()

    def reset_recorders(self):
        raise NotImplementedError()



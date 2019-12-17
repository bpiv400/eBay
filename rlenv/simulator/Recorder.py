from statistics import mean
import pandas as pd
import numpy as np
from compress_pickle import dump
from rlenv.env_consts import START_DAY, VERBOSE, START_PRICE, TIME_FEATS
# TODO: MOVE?
SIM = 'sim'
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
SALE_MEAN = 'price_mean'
SALE_COUNT = 'sale_count'
TRIALS = 'trials'
CUT = 'cut'

OFFER_COLS = [LSTG, SIM, THREAD, INDEX, CLOCK, CON, NORM, MESSAGE]
OFFER_COLS = OFFER_COLS + TIME_FEATS
THREAD_COLS = [LSTG, SIM, THREAD, BYR_HIST, CLOCK]
SALE_COLS = [LSTG, SIM, SALE, DUR, PRICE]
VAL_COLS = [LSTG, VAL, SE, SALE_MEAN, SALE_COUNT, TRIALS, CUT]

class Recorder:
    def __init__(self, records_path):
        self.records_path = records_path
        self.sim = -1
        self.lstg = 0
        self.start_time = None
        self.start_price = None

    def update_lstg(self, lookup=None, lstg=None):
        """
        Resets lstg that the recorder is tracking data about
        :param pd.Series lookup: lstg meta data
        :param int lstg: listing id
        """
        self.lstg = lstg
        self.start_time = lookup[START_DAY]
        self.sim = -1
        self.start_price = lookup[START_PRICE]

    def reset_sim(self):
        self.sim += 1

    def dump(self, recorder_count):
        self.records2frames()
        self.compress_frames()
        output_path = '{}{}.gz'.format(self.records_path, recorder_count)
        output = self.construct_output()
        dump(output, output_path)
        self.reset_recorders()

    def print_offer(self, event, summary):
        """
        Prints data about the offer if verbose
        """
        if VERBOSE:
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
                if delay == 0:
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

    @staticmethod
    def record2frame(record, cols):
        """
        Converts the given list of lists into a dataframe with the given column names
        :param [[numeric]] record:
        :param [str] cols:
        :return: pd.DataFrame with columns given by cols
        """
        record = pd.DataFrame(data=record, columns=cols)
        return record

    @staticmethod
    def print_sale(sale, price, dur):
        """
        Print info about the given sale data if VERBOSE
        """
        if VERBOSE:
            if sale:
                print('Item sold for: {} in {} seconds'.format(price, dur))
            else:
                print('Item did not sell')

    def start_thread(self, thread_id=None, time=None, byr_hist=None):
        raise NotImplementedError()

    def add_offer(self, event, time_feats):
        raise NotImplementedError()

    def add_sale(self, sale, price, dur):
        raise NotImplementedError()

    def records2frames(self):
        raise NotImplementedError()

    def construct_output(self):
        raise NotImplementedError()

    def compress_frames(self):
        raise NotImplementedError()

    def reset_recorders(self):
        raise NotImplementedError()


class DiscrimRecorder(Recorder):
    def __init__(self, records_path):
        super(DiscrimRecorder, self).__init__(records_path)
        # tracker dictionaries
        self.offers = []
        self.values = []
        self.threads = []
        self.sales = []

    def reset_recorders(self):
        self.offers = []
        self.threads = []
        self.values = []
        self.sales = []

    @staticmethod
    def init_dict(cols):
        curr_dict = dict()
        for col in cols:
            curr_dict[col] = list()
        return curr_dict

    def start_thread(self, thread_id=None, byr_hist=None, time=None):
        """
        Records an arrival
        :param thread_id: int giving the thread id
        :param byr_hist: float giving byr history decile
        :param int time: time of the arrival period / offer # TODO: FIGURE OUT FROM ETAN
        """
        byr_hist = int(10 * byr_hist)
        # change ordering if THREAD_COLS ordering changes
        row = [self.lstg, self.sim, thread_id, byr_hist, time]
        self.threads.append(row)

    def add_offer(self, event, time_feats):
        # change ordering if OFFER_COLS changes
        summary = event.summary()
        con, norm, msg, split = summary
        row = [self.lstg,
               self.sim,
               event.thread_id,
               event.turn,
               event.priority,
               con,
               norm,
               msg
               ]
        row = row + list(time_feats)
        self.offers.append(row)
        self.print_offer(event, summary)

    def add_sale(self, sale, price, dur):
        row = [
            self.lstg,
            self.sim,
            sale,
            dur,
            price
        ]
        self.sales.append(row)
        self.print_sale(sale, price, dur)

    @staticmethod
    def compress_offers(offers):
        """
        Compress offers dataframe to smallest possible representation
        :param pd.DataFrame offers: info about each offer
        :return: compressed pd.DataFrame
        """
        offers[CLOCK] = offers[CLOCK].astype(np.int32)
        offers[CON] = offers[CON].astype(np.uint8)
        offers[NORM] = offers[NORM].astype(np.uint8)
        offers[MESSAGE] = offers[MESSAGE].astype(bool)
        offers[INDEX] = offers[INDEX].astype(np.uint8)
        return offers

    @staticmethod
    def compress_sales(sales):
        sales[DUR] = sales[DUR].astype(np.int32)
        sales[SALE] = sales[SALE].astype(bool)
        sales[PRICE] = sales[PRICE].astype(np.float32)
        return sales

    @staticmethod
    def compress_threads(threads):
        threads[BYR_HIST] = threads[BYR_HIST].astype(np.uint8)
        threads[CLOCK] = threads[CLOCK].astype(np.int32)
        return threads

    @staticmethod
    def compress_common_cols(cols, frames, dtypes):
        for i, col in enumerate(cols):
            for frame in frames:
                frame[col] = frame[col].astype(dtypes[i])

    def compress_frames(self):
        # maximally compress offers datatypes
        self.offers = self.compress_offers(self.offers)
        self.sales = self.compress_sales(self.sales)
        self.threads = self.compress_threads(self.threads)

        self.compress_common_cols([LSTG, SIM], [self.threads, self.offers, self.sales],
                                  [np.int32, np.uint16])
        self.compress_common_cols([THREAD], [self.threads, self.offers], [np.uint16])

    def records2frames(self):
        # convert all three dictionaries to dataframes
        self.sales = self.record2frame(self.sales, SALE_COLS)
        self.offers = self.record2frame(self.offers, OFFER_COLS)
        self.threads = self.record2frame(self.threads, THREAD_COLS)

    def construct_output(self):
        records = {
            'offers': self.offers,
            'threads': self.threads,
            'sales': self.sales
        }
        return records


class ValueRecorder(Recorder):
    def __init__(self, record_path):
        super(Recorder, self).__init__(record_path)
        self.values = []

    def start_thread(self, thread_id=None, time=None, byr_hist=None):
        pass

    def add_offer(self, event, time_feats):
        if VERBOSE:
            self.print_offer(event, event.summary())

    def add_sale(self, sale, price, dur):
        self.print_sale(sale, price, dur)

    def records2frames(self):
        self.values = self.record2frame(self.values, VAL_COLS)

    def construct_output(self):
        return self.values

    def compress_frames(self):
        self.values = self.compress_values(self.values)

    def reset_recorders(self):
        self.values = []

    def add_val(self, val_calc):
        """
        Records statistics related to the value of the current lstg
        :param val_calc: value calculator object for current lstg
        :type val_calc: rlenv.ValueCalculator.ValueCalculator
        """
        # reorder if VAL_COLS changes
        row = [
            self.lstg,
            val_calc.mean,
            val_calc.mean_se,
            mean(val_calc.sales),
            len(val_calc.sales),
            val_calc.exp_count,
            val_calc.cut
        ]
        self.values.append(row)

    @staticmethod
    def compress_values(values):
        """
        Compresses values dataframe
        """
        for col in [SALE_MEAN, VAL, SE, CUT]:
            values[col] = values[col].astype(np.float32)
        values[LSTG] = values[LSTG].astype(np.int32)
        values[SALE_COUNT] = values[SALE_COUNT].astype(np.int32)
        values[TRIALS] = values[TRIALS].astype(np.uint16)
        return values


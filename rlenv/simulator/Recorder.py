from statistics import mean
import pandas as pd
import numpy as np
from compress_pickle import dump
from rlenv.env_consts import START_DAY, VERBOSE, START_PRICE

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

OFFER_COLS = [SIM, INDEX, THREAD, CLOCK, CON, NORM, MESSAGE, LSTG]
THREAD_COLS = [SIM, THREAD, BYR_HIST, LSTG]
SALE_COLS = [SIM, SALE, DUR, PRICE, LSTG]
VAL_COLS = [LSTG, VAL, SE, SALE_MEAN, SALE_COUNT, TRIALS, CUT]


class Recorder:
    def __init__(self, records_path, record_values=False):
        self.records_path = records_path
        self.record_values = record_values

        # tracker dictionaries
        self.offers = None
        self.values = None
        self.threads = None
        self.sales = None
        self.reset_recorders()

        # lstg feats
        self.lstg = None
        self.start_time = None
        self.sim = None
        self.start_price = None

    def reset_recorders(self):
        self.offers = self.init_dict(OFFER_COLS)
        self.threads = self.init_dict(THREAD_COLS)
        self.values = self.init_dict(VAL_COLS)
        self.sales = self.init_dict(SALE_COLS)

    @staticmethod
    def init_dict(cols):
        curr_dict = dict()
        for col in cols:
            curr_dict[col] = list()
        return curr_dict

    def update_lstg(self, lookup=None, lstg=None):
        self.lstg = lstg
        self.start_time = lookup[START_DAY]
        self.sim = -1
        self.start_price = lookup[START_PRICE]

    def start_thread(self, thread_id=None, byr_hist=None):
        byr_hist = int(10 * byr_hist)
        self.threads[LSTG].append(self.lstg)
        self.threads[SIM].append(self.sim)
        self.threads[THREAD].append(thread_id)
        self.threads[BYR_HIST].append(byr_hist)

    def reset_sim(self):
        self.sim += 1

    def add_offer(self, event):
        self.offers[LSTG].append(self.lstg)
        self.offers[SIM].append(self.sim)
        self.offers[INDEX].append(event.turn)
        self.offers[THREAD].append(event.thread_id)
        self.offers[CLOCK].append(event.priority)
        summary = event.summary()
        con, norm, msg, split = summary
        self.offers[CON].append(con)
        self.offers[NORM].append(norm)
        self.offers[MESSAGE].append(msg)
        if VERBOSE:
            self.print_offer(event, summary)

    def print_offer(self, event, summary):
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

    def add_sale(self, sale, price, dur):
        self.sales[LSTG].append(self.lstg)
        self.sales[SALE].append(sale)
        self.sales[PRICE].append(price)
        self.sales[DUR].append(dur)
        self.sales[SIM].append(self.sim)
        if VERBOSE:
            if sale:
                print('Item sold for: {}'.format(price))
            else:
                print('Item did not sell')

    def add_val(self, val_calc):
        """
        Records statistics related to the value of the current lstg
        :param val_calc: value calculator object for current lstg
        :type val_calc: rlenv.ValueCalculator.ValueCalculator
        :return: None
        """
        self.values[LSTG].append(self.lstg)
        self.values[VAL].append(val_calc.mean)
        self.values[SE].append(val_calc.mean_se)
        self.values[TRIALS].append(val_calc.exp_count)
        self.values[SALE_MEAN].append(mean(val_calc.sales))
        self.values[CUT].append(val_calc.cut)
        self.values[SALE_COUNT].append(len(val_calc.sales))

    def compress_offers(self):
        self.offers[CLOCK] = self.offers[CLOCK].astype(np.int32)
        self.offers[CON] = self.offers[CON].astype(np.uint8)
        self.offers[NORM] = self.offers[NORM].astype(np.uint8)
        self.offers[MESSAGE] = self.offers[MESSAGE].astype(bool)
        self.offers[INDEX] = self.offers[INDEX].astype(np.uint8)

    def compress_sales(self):
        self.sales[DUR] = self.sales[DUR].astype(np.int32)
        self.sales[SALE] = self.sales[SALE].astype(bool)
        self.sales[PRICE] = self.sales[PRICE].astype(np.float32)

    def compress_threads(self):
        self.threads[BYR_HIST] = self.threads[BYR_HIST].astype(np.uint8)

    def compress_values(self):
        if self.record_values:
            self.values[VAL] = self.values[VAL].astype(np.float32)
            self.values[SE] = self.values[SE].astype(np.float32)
            self.values[SALE_COUNT] = self.values[SALE_COUNT].astype(np.uint16)
            self.values[TRIALS] = self.values[TRIALS].astype(np.uint16)
            self.values[CUT] = self.values[CUT].astype(np.float32)

    @staticmethod
    def compress_common_cols(cols, frames, dtypes):
        for i, col in enumerate(cols):
            for frame in frames:
                frame[col] = frame[col].astype(dtypes[i])

    def compress_records(self):
        # maximally compress offers datatypes
        self.compress_offers()
        self.compress_sales()
        self.compress_threads()
        self.compress_values()
        self.compress_common_cols([SIM],
                                  [self.threads, self.offers, self.sales],
                                  [np.uint16])
        self.compress_common_cols([THREAD], [self.threads, self.offers], [np.uint16])
        if self.record_values:
            frames = [self.threads, self.offers, self.sales, self.values]
        else:
            frames = [self.threads, self.offers, self.sales]
        self.compress_common_cols([LSTG], frames, [np.int32])

    def dump(self, recorder_count):
        # convert all three dictionaries to dataframes
        self.sales = pd.DataFrame.from_dict(self.sales)
        self.offers = pd.DataFrame.from_dict(self.offers)
        self.threads = pd.DataFrame.from_dict(self.threads)
        if self.record_values:
            self.values = pd.DataFrame.from_dict(self.values)

        #compress
        self.compress_records()

        records_path = '{}{}_records.gz'.format(self.records_path, recorder_count)
        records = {
            'offers': self.offers,
            'threads': self.threads,
            'sales': self.sales
        }
        dump(records, records_path)

        if self.record_values:
            values_path = '{}{}_vals.gz'.format(self.records_path, recorder_count)
            dump(self.values, values_path)
        self.reset_recorders()


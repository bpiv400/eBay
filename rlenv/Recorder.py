import pandas as pd
import numpy as np
from compress_pickle import dump
from rlenv.env_consts import START_DAY, VERBOSE, START_PRICE, PRICE
from rlenv.env_utils import chunk_dir, get_cut
from rlenv.ValueCalculator import ValueCalculator

# TODO: MOVE?
SIM = 'sim'
INDEX = 'index'
VAL = 'val'
THREAD = 'thread'
CLOCK = 'clock'
BYR_HIST = 'byr_hist'
NORM = 'norm'
MESSAGE = 'message'
CON = 'con'
DUR = 'duration'
SALE = 'sale'
LSTG = 'lstg'
SE = 'SE'

OFFER_COLS = [SIM, INDEX, THREAD, CLOCK, CON, NORM, MESSAGE, LSTG]
THREAD_COLS = [SIM, THREAD, BYR_HIST, LSTG]
VAL_COLS = [LSTG, VAL, SE]


class Recorder:
    def __init__(self, chunk=None):
        self.chunk = chunk

        # tracker dictionaries
        self.offers = dict()
        self.threads = dict()
        self.values = dict()

        # lstg feats
        self.lstg = None
        self.start_time = None
        self.sim = None
        self.start_price = None

        for col in OFFER_COLS:
            self.offers[col] = list()
        for col in THREAD_COLS:
            self.threads[col] = list()
        for col in VAL_COLS:
            self.sales[col] = list()

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

    # probably need to change this
    def reset_sim(self):
        self.sim += 1

    def add_offer(self, event):
        self.offers[LSTG].append(self.lstg)
        self.offers[SIM].append(self.sim)
        self.offers[INDEX].append(event.turn)
        self.offers[THREAD].append(event.thread_id)
        self.offers[CLOCK].append(event.priority)
        con, norm, msg, split = event.summary()
        self.offers[CON].append(con)
        self.offers[NORM].append(norm)
        self.offers[MESSAGE].append(msg)
        if event.turn > 1:
            days, delay = event.delay_outcomes()
        else:
            days, delay = 0, 0
        byr = event.turn % 2 != 0
        if VERBOSE:
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

    def add_sale(self):
        raise RuntimeError("not implemented -- make sure add sale adds the offer")

    def add_val(self, val):
        self.sales[LSTG].append(self.lstg)
        self.sales[VAL].append(val)

    def dump(self, base_dir, recorder_count):
        # convert all three dictionaries to dataframes
        self.sales = pd.DataFrame.from_dict(self.sales)
        self.offers = pd.DataFrame.from_dict(self.offers)
        self.threads = pd.DataFrame.from_dict(self.threads)
        # maximally compress offers datatypes
        self.offers[LSTG] = self.offers[LSTG].astype(np.int32)
        self.offers[CLOCK] = self.offers[CLOCK].astype(np.int32)
        self.offers[CON] = self.offers[CON].astype(np.uint8)
        self.offers[NORM] = self.offers[NORM].astype(np.uint8)
        self.offers[MESSAGE] = self.offers[MESSAGE].astype(bool)
        self.offers[INDEX] = self.offers[INDEX].astype(np.uint8)
        for col in [SIM, THREAD]:
            for frame in [self.threads, self.offers]:
                frame[col] = frame[col].astype(np.uint16)
        # maximally compress threads dataframe
        self.threads[BYR_HIST] = self.threads[BYR_HIST].astype(np.uint8)
        self.threads[LSTG] = self.threads[LSTG].astype(np.int32)

        # maximally compress  sales
        self.sales[DUR] = self.sales[DUR].astype(np.int32)
        self.sales[SALE] = self.sales[SALE].astype(bool)
        self.sales[REWARD] = self.sales[REWARD].astype(np.float32)
        self.sales[LSTG] = self.sales[LSTG].astype(np.int32)

        records_path = '{}{}.gz'.format(chunk_dir(base_dir, self.chunk, records=True),
                                        recorder_count)
        rewards_path = '{}{}.gz'.format(chunk_dir(base_dir, self.chunk, rewards=True),
                                        recorder_count)
        records = {
            'offers': self.offers,
            'threads': self.threads
        }
        dump(records, records_path)
        dump(self.sales, rewards_path)
        self.sales = dict()
        self.offers = dict()
        self.threads = dict()

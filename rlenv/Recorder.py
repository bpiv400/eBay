import pandas as pd
import numpy as np
from compress_pickle import dump
from env_consts import START_DAY, INTERACT

SIM = 'sim'
INDEX = 'index'
THREAD = 'thread'
CLOCK = 'clock'
BYR_HIST = 'byr_hist'
NORM = 'norm'
MESSAGE = 'message'
CON = 'con'
REWARD = 'reward'
DUR = 'duration'
SALE = 'sale'

OFFER_COLS = [SIM, INDEX, THREAD, CLOCK, CON, NORM, MESSAGE]
THREAD_COLS = [SIM, THREAD, BYR_HIST]
SALE_COLS = [SIM, SALE, DUR, REWARD]


class Recorder:
    def __init__(self, lstg, lookup):
        self.lstg = lstg
        self.start_time = lookup[START_DAY]
        self.sim = -1
        self.offers = dict()
        self.threads = dict()
        self.sales = dict()

        for col in OFFER_COLS:
            self.offers[col] = list()
        for col in THREAD_COLS:
            self.threads[col] = list()
        for col in SALE_COLS:
            self.sales[col] = list()

    def start_thread(self, thread_id=None, byr_hist=None):
        byr_hist = int(10 * byr_hist)
        self.threads[SIM].append(self.sim)
        self.threads[THREAD].append(thread_id)
        self.threads[BYR_HIST].append(byr_hist)

    def reset(self):
        self.sim += 1

    def add_offer(self, event):
        self.offers[SIM].append(self.sim)
        self.offers[INDEX].append(event.turn)
        self.offers[THREAD].append(event.thread_id)
        self.offers[CLOCK].append(event.priority)
        con, norm, msg, split = event.summary()
        self.offers[CON].append(con)
        self.offers[NORM].append(norm)
        self.offers[MESSAGE].append(msg)
        days, delay = event.delay_outcomes()

        if INTERACT:
            if con == 0:
                if delay == 1:
                    otype = 'expiration rejection'
                elif delay == 0:
                    otype = 'automatic rejection'
                else:
                    otype = 'standard rejection'
            elif con == 1:
                if delay == 0:
                    otype = 'automatic acceptance'
                else:
                    otype = 'standard acceptance'
            else:
                otype = 'standard offer'
            print('Thread: {} | Offer: {} | clock: {}'.format(event.thread_id,
                                                              event.turn, event.priority))
            print('Days: {}, Delay: {}').format()
            print('Offer type: {}'.format(otype))
            print('Concession: {} | start price norm: {} | split: {} | message: {}'.format(
                con, norm, split, msg))
            if event.turn % 2 == 0:
                auto, exp, rej = event.slr_outcomes()
                print('Auto: {} | Exp: {} | Reject: {}'.format(auto, exp, rej))

    def add_sale(self, sale, reward, dur):
        self.sales[SALE].append(sale)
        self.sales[REWARD].append(reward)
        self.sales[DUR].append(dur)
        if INTERACT:
            if sale:
                print('Item sold w/ reward: {}'.format(reward))
            else:
                print('Item did not sell')

    def dump(self, base_dir):
        # convert all three dictionaries to dataframes
        self.sales = pd.from_dict(self.sales)
        self.offers = pd.from_dict(self.offers)
        self.threads = pd.from_dict(self.threads)
        # maximally compress offers datatypes
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
        # maximally compress  sales
        self.sales[DUR] = self.sales[DUR].astype(np.int32)
        self.sales[SALE] = self.sales[SALE].astype(bool)
        self.sales[REWARD] = self.sales[REWARD].astype(np.float32)

        records_path = '{}records/{}.gz'.format(base_dir, self.lstg)
        rewards_path = '{}rewards/{}.gz'.format(base_dir, self.lstg)
        records = {
            'offers': self.offers,
            'threads': self.threads
        }
        dump(records, records_path)
        dump(self.sales, rewards_path)

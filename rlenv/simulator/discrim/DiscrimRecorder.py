import numpy as np
from rlenv.simulator.Recorder import *


class DiscrimRecorder(Recorder):
    def __init__(self, records_path=None, verbose=None, record_sim=False):
        super(DiscrimRecorder, self).__init__(records_path, verbose)
        self.offers = None
        self.threads = None
        self.record_sim = record_sim
        self.reset_recorders()

    def reset_recorders(self):
        # tracker dictionaries
        self.offers = []
        self.threads = []

    def start_thread(self, thread_id=None, byr_hist=None, time=None):
        """
        Records an arrival
        :param thread_id: int giving the thread id
        :param byr_hist: float giving byr history decile
        :param int time: time of the offer
        """
        byr_hist = int(10 * byr_hist)
        row = [self.lstg, thread_id, byr_hist, time]
        if self.record_sim:
            row.append(self.sim)
        self.threads.append(row)

    def add_offer(self, event=None, time_feats=None, censored=False):
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
               msg,
               censored
               ]
        row += list(time_feats)
        if self.record_sim:
            row.append(self.sim)
        self.offers.append(row)
        if self.verbose:
            if censored:
                print('censored')
            self.print_offer(event)

    @staticmethod
    def compress_common_cols(cols, frames, dtypes):
        for i, col in enumerate(cols):
            for frame in frames:
                frame[col] = frame[col].astype(dtypes[i])

    def compress_frames(self):
        # offers dataframe
        self.offers[INDEX] = self.offers[INDEX].astype(np.uint8)
        self.offers[CON] = self.offers[CON].astype(np.uint8)
        self.offers[MSG] = self.offers[MSG].astype(bool)
        self.offers[CENSOR] = self.offers[CENSOR].astype(bool)
        for name in TIME_FEATS:
            if 'offers' in name or 'count' in name:
                self.offers[name] = self.offers[name].astype(np.uint8)

        # threads dataframe
        self.threads[BYR_HIST] = self.threads[BYR_HIST].astype(np.uint8)

        # common columns in offers and threads dataframes
        self.compress_common_cols([LSTG, THREAD, CLOCK],
                                  [self.threads, self.offers],
                                  [np.int32, np.uint16, np.int32])
        self.offers.set_index(INDEX_COLS[self.record_sim], inplace=True)
        self.threads.set_index(INDEX_COLS[self.record_sim][:-1], inplace=True)
        assert np.all(self.offers.xs(1, level='index')[CON] > 0)

    def records2frames(self):
        # convert both dictionaries to dataframes
        offer_cols = OFFER_COLS
        thread_cols = THREAD_COLS
        if self.record_sim:
            offer_cols += ['sim']
            thread_cols += ['sim']
        self.offers = self.record2frame(self.offers, offer_cols)
        self.threads = self.record2frame(self.threads, thread_cols)

    def construct_output(self):
        return {'offers': self.offers.sort_index(),
                'threads': self.threads.sort_index()}

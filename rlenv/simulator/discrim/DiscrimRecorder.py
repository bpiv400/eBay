import numpy as np
from rlenv.simulator.Recorder import *


class DiscrimRecorder(Recorder):
    def __init__(self, records_path):
        super(DiscrimRecorder, self).__init__(records_path)
        self.reset_recorders()

    def reset_recorders(self):
        # tracker dictionaries
        self.offers = []
        self.threads = []

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
        :param int time: time of the offer
        """
        byr_hist = int(10 * byr_hist)
        # change ordering if THREAD_COLS ordering changes
        row = [self.lstg, self.sim, thread_id, byr_hist, time]
        self.threads.append(row)

    def add_offer(self, event, time_feats):
        # change ordering if OFFER_COLS changes
        summary = event.summary()
        con, _, msg, _ = summary
        row = [self.lstg,
               self.sim,
               event.thread_id,
               event.turn,
               event.priority,
               con,
               msg
               ]
        row = row + list(time_feats)
        self.offers.append(row)
        self.print_offer(event, summary)

    @staticmethod
    def compress_offers(offers):
        """
        Compress offers dataframe to smallest possible representation
        :param pd.DataFrame offers: info about each offer
        :return: compressed pd.DataFrame
        """
        offers[CLOCK] = offers[CLOCK].astype(np.int32)
        offers[CON] = offers[CON].astype(np.uint8)
        offers[MESSAGE] = offers[MESSAGE].astype(bool)
        offers[INDEX] = offers[INDEX].astype(np.uint8)
        return offers

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
        self.threads = self.compress_threads(self.threads)
        self.compress_common_cols([LSTG, SIM, THREAD], 
                                  [self.threads, self.offers],
                                  [np.int32, np.uint16, np.uint16])

    def records2frames(self):
        # convert all three dictionaries to dataframes
        self.offers = self.record2frame(self.offers, OFFER_COLS)
        self.threads = self.record2frame(self.threads, THREAD_COLS)

    def construct_output(self):
        return {'offers': self.offers, 'threads': self.threads}
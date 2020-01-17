import numpy as np
from rlenv.simulator.Recorder import *


class ValueRecorder(Recorder):
    def __init__(self, record_path, verbose):
        super().__init__(record_path, verbose)
        self.values = []

    def start_thread(self, thread_id=None, time=None, byr_hist=None):
        pass

    def add_offer(self, event, time_feats=None, censored=False):
        if self.verbose:
            self.print_offer(event, event.summary())

    def records2frames(self):
        self.values = self.record2frame(self.values, VAL_COLS)

    def construct_output(self):
        return self.values

    def compress_frames(self):
        for col in [AVG_PRICE, VAL, SE, CUT, P_SALE]:
            self.values[col] = self.values[col].astype(np.float32)
        self.values[LSTG] = self.values[LSTG].astype(np.int32)
        self.values[NUM_SALES] = self.values[NUM_SALES].astype(np.int32)
        self.values.set_index('lstg', inplace=True)

    def reset_recorders(self):
        self.values = []

    def add_val(self, val_calc):
        """
        Records statistics related to the value of the current lstg
        :param val_calc: rlenv.ValueCalculator.ValueCalculator object for current lstg
        """
        # VAL_COLS = [LSTG, VAL, SE, AVG_PRICE, NUM_SALES, P_SALE, CUT]
        row = [
            self.lstg,
            val_calc.value,
            val_calc.se,
            val_calc.mean_x,
            val_calc.num_sales,
            val_calc.p_sale,
            val_calc.cut
        ]
        self.values.append(row)

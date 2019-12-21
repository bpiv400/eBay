from statistics import mean
import numpy as np
from rlenv.simulator.Recorder import *


class ValueRecorder(Recorder):
    def __init__(self, record_path, verbose):
        super( ).__init__(record_path, verbose)
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

    def dump(self, recorder_count):
        self.records2frames()
        self.compress_frames()
        output_path = '{}{}.gz'.format(self.records_path, recorder_count)
        output = self.construct_output()
        dump(output, output_path)
        self.reset_recorders()

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


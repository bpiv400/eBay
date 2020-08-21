from featnames import LSTG
from rlenv.generate.Recorder import Recorder

VAL_COLS = [LSTG, 'value', 'se', 'avg_price', 'num_sim', 'num_sales']


class ValueRecorder(Recorder):

    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)
        self.values = []

    def start_thread(self, thread_id=None, time=None, byr_hist=None):
        pass

    def add_offer(self, event, time_feats=None, censored=False):
        if self.verbose:
            self.print_offer(event)

    def construct_output(self):
        self.values = self.record2frame(self.values, VAL_COLS)
        self.values.set_index('lstg', inplace=True)
        return self.values

    def reset_recorders(self):
        self.values = []

    def add_val(self, val_calc):
        """
        Records statistics related to the value of the current lstg
        :param val_calc: rlenv.ValueCalculator.ValueCalculator object for current lstg
        """
        row = [
            int(self.lstg),
            val_calc.value,
            val_calc.se,
            val_calc.mean_x,
            val_calc.ct,
            val_calc.sales
        ]
        self.values.append(row)
        print('{0:d}: {1:d} simulations, {2:2.1f}%'.format(
            row[0], row[4], 100 * row[3] / val_calc.start_price))

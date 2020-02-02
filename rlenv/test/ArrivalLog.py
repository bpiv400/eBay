from rlenv.env_utils import compare_input_dicts
from rlenv.env_consts import ARRIVAL_MODEL, BYR_HIST_MODEL


class ArrivalLog:

    def __init__(self, hist=None, time=None, arrival_inputs=None, hist_inputs=None, check_time=None):
        self.censored = hist is None
        self.time = time
        self.arrival_inputs = arrival_inputs
        self.hist_inputs = hist_inputs
        self.hist = hist
        self.check_time = check_time

    def get_inter_arrival(self, check_time=None, input_dict=None):
        assert check_time == self.check_time
        compare_input_dicts(model=ARRIVAL_MODEL, stored_inputs=self.arrival_inputs, env_inputs=input_dict)
        return self.time - self.check_time

    def get_hist(self, check_time=None, input_dict=None):
        if self.censored:
            raise RuntimeError("Checking history for censored arrival event")
        assert check_time == self.time
        compare_input_dicts(model=BYR_HIST_MODEL, stored_inputs=self.hist_inputs, env_inputs=input_dict)
        return self.hist

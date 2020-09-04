from constants import BYR_HIST_MODEL, INTERARRIVAL_MODEL
from testing.util import compare_input_dicts
from featnames import LSTG


class Arrival:

    def __init__(
            self,
            hist=None,
            time=None,
            arrival_inputs=None,
            hist_inputs=None,
            check_time=None,
            first_arrival=False,
            agent_buyer=False
    ):
        self.censored = hist is None
        self.time = time
        self.first_arrival = first_arrival
        self.arrival_inputs = arrival_inputs
        self.hist_inputs = hist_inputs
        self.hist = hist
        self.check_time = check_time
        self.agent_buyer = agent_buyer

    def get_inter_arrival(self, check_time=None, input_dict=None):
        assert check_time == self.check_time
        if input_dict is not None:
            # days since last should ignore buyer agent; a workaround
            if self.agent_buyer:
                input_dict = input_dict.copy()
                input_dict[LSTG][:, -2] = self.arrival_inputs[LSTG][:, -2]

            compare_input_dicts(model=INTERARRIVAL_MODEL,
                                stored_inputs=self.arrival_inputs,
                                env_inputs=input_dict)
        return self.inter_arrival

    @property
    def inter_arrival(self):
        inter_arrival = self.time - self.check_time
        return int(inter_arrival)

    def get_hist(self, check_time=None, input_dict=None):
        if self.censored:
            raise RuntimeError("Checking history for censored arrival event")
        assert check_time == self.time
        compare_input_dicts(model=BYR_HIST_MODEL,
                            stored_inputs=self.hist_inputs,
                            env_inputs=input_dict)
        return self.hist

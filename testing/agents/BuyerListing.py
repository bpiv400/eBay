from featnames import START_TIME
from constants import MAX_DELAY_ARRIVAL
from testing.agents.AgentListing import AgentListing


class BuyerListing(AgentListing):

    def __init__(self, params=None):
        """
        :param params: dict
        """
        self.agent_thread = params['thread_id']
        super().__init__(params=params, byr=True)
        self.update_arrival_time()

    def _push_actions(self, params=None):
        full_inputs = params['inputs'][self.actions.model_name]
        agent_turns = self.threads[self.agent_thread].get_agent_turns()
        self.record_thread(turns=agent_turns,
                           thread_id=self.agent_thread,
                           full_inputs=full_inputs)

    def update_arrival_time(self):
        if (self.agent_thread + 1) in self.arrivals:
            target_time = self.arrivals[self.agent_thread + 1].time
        else:
            target_time = self.lookup[START_TIME] + MAX_DELAY_ARRIVAL
        target_censored = target_time == (self.lookup[START_TIME] + MAX_DELAY_ARRIVAL)
        self.arrivals[self.agent_thread].time = target_time
        self.arrivals[self.agent_thread].censored = target_censored

    def get_agent_arrival(self):
        return self.actions.get_buyer_arrival() - self.start_time

    def is_agent_arrival(self, thread_id=None):
        return thread_id == self.agent_thread

    def is_agent_thread(self, thread_id=None):
        return thread_id == self.agent_thread

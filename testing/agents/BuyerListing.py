from featnames import BYR
from testing.agents.AgentListing import AgentListing


class BuyerListing(AgentListing):

    def __init__(self, params=None):
        """
        :param params: dict
        """
        self.agent_thread = params['thread_id']
        super().__init__(params=params, byr=True)

    def _push_actions(self, params=None):
        full_inputs = params['inputs'][BYR]
        if self.agent_thread in self.threads:
            agent_turns = self.threads[self.agent_thread].get_agent_turns()
        else:
            agent_turns = dict()
        self.record_thread(turns=agent_turns,
                           thread_id=self.agent_thread,
                           full_inputs=full_inputs)

    def get_agent_arrival(self):
        return self.actions.get_buyer_arrival() - self.start_time

    def is_agent_arrival(self, thread_id=None):
        return thread_id == self.agent_thread

    def is_agent_thread(self, thread_id=None):
        return thread_id == self.agent_thread

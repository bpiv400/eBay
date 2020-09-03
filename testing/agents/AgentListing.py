from testing.agents.ActionQueue import ActionQueue
from testing.Listing import Listing
from testing.Thread import Thread


class AgentListing(Listing):
    def __init__(self, params=None, byr=False):
        self.byr = byr
        super().__init__(params=params)
        self.actions = ActionQueue(byr=self.byr)
        self._push_actions(params)

    def generate_thread(self, thread_id=None, params=None):
        thread_params = self._get_thread_params(thread_id=thread_id,
                                                params=params)
        agent_thread = self.is_agent_thread(thread_id=thread_id)
        return Thread(params=thread_params,
                      arrival_time=self.arrivals[thread_id].time,
                      agent=agent_thread,
                      agent_buyer=self.byr)

    def get_action(self, agent_tuple=None):
        return self.actions.get_action(agent_tuple=agent_tuple)

    def verify_done(self):
        self.actions.verify_done()

    def _push_actions(self, params):
        raise NotImplementedError()

    def record_agent_thread(self, turns=None, thread_id=None, full_inputs=None):
        raise NotImplementedError()

    def is_agent_thread(self, thread_id=None):
        raise NotImplementedError()

from testing.agents.ActionQueue import ActionQueue
from testing.Listing import Listing


class AgentListing(Listing):
    def __init__(self, params=None, byr=False):
        self.byr = byr
        super().__init__(params=params)
        self.actions = ActionQueue(byr=byr)
        self._push_actions(params)

    def get_action(self, agent_tuple=None):
        return self.actions.get_agent_con(agent_tuple=agent_tuple)

    def verify_done(self):
        self.actions.verify_done()

    def _push_actions(self, params):
        raise NotImplementedError()

    @property
    def is_agent_buyer(self):
        return self.byr

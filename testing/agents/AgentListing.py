from testing.agents.Action import Action
from testing.agents.ActionQueue import ActionQueue
from testing.Listing import Listing
from testing.util import populate_inputs


class AgentListing(Listing):
    def __init__(self, params=None, byr=False):
        self.byr = byr
        super().__init__(params=params)
        self.actions = ActionQueue(byr=byr)
        self._push_actions(params)

    def record_thread(self, turns=None, thread_id=None, full_inputs=None):
        for t, turn in turns.items():
            index = (thread_id, t)
            if turn.is_censored:
                input_dict = None
            else:
                input_dict = populate_inputs(full_inputs=full_inputs,
                                             value=index,
                                             agent=True,
                                             agent_byr=self.byr)
            action = Action(con=turn.agent_con(),
                            censored=turn.is_censored,
                            time=turn.agent_time(),
                            input_dict=input_dict,
                            turn=t,
                            thread_id=thread_id)
            self.actions.push_action(action=action)

    def get_action(self, agent_tuple=None):
        return self.actions.get_agent_con(agent_tuple=agent_tuple)

    def verify_done(self):
        self.actions.verify_done()

    def _push_actions(self, params):
        raise NotImplementedError()

    @property
    def is_agent_buyer(self):
        return self.byr

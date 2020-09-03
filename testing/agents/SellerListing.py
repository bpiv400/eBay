from testing.Action import Action
from testing.agents.AgentListing import AgentListing
from testing.util import populate_inputs


class SellerListing(AgentListing):
    def __init__(self, params):
        super().__init__(params=params, byr=False)

    def is_agent_thread(self, thread_id=None):
        return True

    def record_agent_thread(self, turns=None, thread_id=None, full_inputs=None):
        for t, turn in turns.items():
            time = turn.agent_time()
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
                            time=time,
                            input_dict=input_dict,
                            turn=t,
                            thread_id=thread_id)
            self.actions.push_action(action=action)

    def _push_actions(self, params):
        full_inputs = params['inputs'][self.actions.model_name]
        for thread_id, thread in self.threads.items():
            agent_turns = thread.get_agent_turns()
            if len(agent_turns) != 0:
                self.record_agent_thread(full_inputs=full_inputs,
                                         thread_id=thread_id,
                                         turns=agent_turns)

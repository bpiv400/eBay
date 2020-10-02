from testing.agents.AgentListing import AgentListing


class SellerListing(AgentListing):
    def __init__(self, params):
        super().__init__(params=params, byr=False)

    def is_agent_thread(self, thread_id=None):
        return True

    def _push_actions(self, params):
        full_inputs = params['inputs'][self.actions.model_name]
        for thread_id, thread in self.threads.items():
            agent_turns = thread.get_agent_turns()
            if len(agent_turns) != 0:
                self.record_thread(full_inputs=full_inputs,
                                   thread_id=thread_id,
                                   turns=agent_turns)

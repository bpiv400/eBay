from testing.agents.ActionQueue import ActionQueue
from testing.Action import Action
from testing.Listing import Listing
from testing.Thread import Thread
from testing.util import populate_inputs


class SellerListing(Listing):
    def __init__(self, params):
        super().__init__(params=params)
        self.agent_log = self.generate_agent_log(params)
        self.byr = False
        self.agent_thread = None

    def generate_agent_log(self, params):
        agent_log = ActionQueue(byr=False)
        full_inputs = params['inputs'][agent_log.model_name]
        for thread_id, thread_log in self.threads.items():
            agent_turns = thread_log.get_agent_turns()
            if len(agent_turns) != 0:
                self.record_agent_thread(agent_log=agent_log,
                                         full_inputs=full_inputs,
                                         thread_id=thread_id,
                                         turns=agent_turns)
        return agent_log

    @staticmethod
    def record_agent_thread(turns=None, agent_log=None, thread_id=None, full_inputs=None):
        for turn_number, turn_log in turns.items():
            time = turn_log.agent_time()
            index = (thread_id, turn_number)
            if turn_log.is_censored:
                input_dict = None
            else:
                input_dict = populate_inputs(full_inputs=full_inputs, value=index,
                                             agent=True,
                                             agent_byr=False)
            action = Action(con=turn_log.agent_con(),
                            censored=turn_log.is_censored,
                            time=time,
                            input_dict=input_dict,
                            turn=turn_number,
                            thread_id=thread_id)
            agent_log.push_action(action=action)

    @property
    def is_agent_arrival(self, thread_id=None):
        return False

    @property
    def is_agent_thread(self, thread_id=None):
        return True

    def generate_thread(self, thread_id=None, params=None):
        thread_params = self._get_thread_params(thread_id=thread_id,
                                                params=params)
        return Thread(params=thread_params,
                      arrival_time=self.arrivals[thread_id].time,
                      agent=True,
                      agent_buyer=False)

    def get_action(self, agent_tuple=None):
        return self.agent_log.get_action(agent_tuple=agent_tuple)

    def verify_done(self):
        self.agent_log.verify_done()

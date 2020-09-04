from featnames import START_TIME, LSTG, DAYS_SINCE_LSTG
from constants import MAX_DELAY_ARRIVAL, DAY
from testing.agents.Action import Action
from testing.agents.AgentListing import AgentListing
from testing.util import populate_inputs


class BuyerListing(AgentListing):

    def __init__(self, params=None):
        """
        :param params: dict
        """
        self.agent_thread = params['thread_id']
        super().__init__(params=params, byr=True)
        self.update_arrival_time()

    def record_buyer_delays(self, full_inputs=None, byr_arrival=None):
        # find the floor of the number of days that have passed since the start of the
        days = int((byr_arrival.time - self.lookup[START_TIME]) / DAY)
        for i in range(days):
            action_index = (self.agent_thread, 1, i)
            input_dict = populate_inputs(full_inputs=full_inputs,
                                         value=action_index,
                                         agent=True,
                                         agent_byr=True)
            delay = full_inputs[LSTG].loc[action_index, DAYS_SINCE_LSTG] * DAY
            time = int(round(self.start_time + delay))
            log = Action(input_dict=input_dict,
                         time=time,
                         con=0,
                         thread_id=self.agent_thread,
                         turn=1)
            self.actions.push_action(action=log)
        return byr_arrival.time

    def record_buyer_thread(self, turns=None, full_inputs=None):
        days = (turns[1].agent_time() - self.lookup[START_TIME]) // DAY
        for t, turn in turns.items():
            index = (self.agent_thread, t, days)
            if turn.is_censored:
                input_dict = None
            else:
                input_dict = populate_inputs(full_inputs=full_inputs,
                                             value=index,
                                             agent=True,
                                             agent_byr=True)
            action = Action(con=turn.agent_con(),
                            censored=turn.is_censored,
                            time=turn.agent_time(),
                            input_dict=input_dict,
                            thread_id=self.agent_thread,
                            turn=t)
            self.actions.push_action(action=action)

    def _push_actions(self, params=None):
        full_inputs = params['inputs'][self.actions.model_name]
        # arrival delays
        byr_arrival = self.generate_arrival(params=params,
                                            thread_id=self.agent_thread)
        self.record_buyer_delays(full_inputs=full_inputs,
                                 byr_arrival=byr_arrival)
        # add offers
        agent_turns = self.threads[self.agent_thread].get_agent_turns()
        self.record_buyer_thread(turns=agent_turns, full_inputs=full_inputs)

    def update_arrival_time(self):
        if (self.agent_thread + 1) in self.arrivals:
            target_time = self.arrivals[self.agent_thread + 1].time
        else:
            target_time = self.lookup[START_TIME] + MAX_DELAY_ARRIVAL
        target_censored = target_time == (self.lookup[START_TIME] + MAX_DELAY_ARRIVAL)
        self.arrivals[self.agent_thread].time = target_time
        self.arrivals[self.agent_thread].censored = target_censored

    def get_agent_arrival(self):
        return self.actions.get_buyer_delay()

    def is_agent_arrival(self, thread_id=None):
        return thread_id == self.agent_thread

    def is_agent_thread(self, thread_id=None):
        return thread_id == self.agent_thread

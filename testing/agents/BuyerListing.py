from featnames import START_TIME, CON, AUTO, X_OFFER, THREAD
from constants import MAX_DELAY_ARRIVAL, DAY
from testing.Action import Action
from testing.agents.AgentListing import AgentListing
from testing.agents.ThreadTranslator import ThreadTranslator
from testing.util import populate_inputs


class BuyerListing(AgentListing):

    def __init__(self, params=None):
        """
        :param params: dict
        """
        super().__init__(params=params, byr=True)
        self.agent_thread = params['thread_id']
        self.update_arrival_time()
        self.translator = ThreadTranslator(agent_thread=self.agent_thread,
                                           arrivals=self.arrivals,
                                           params=params)
        if self.verbose:
            self.translator.print_translator()

    def record_agent_arrivals(self, full_inputs=None):
        byr_arrival = self.arrivals[self.agent_thread]
        # find the floor of the number of days that have passed since the start of the
        days = int((byr_arrival.time - self.lookup[START_TIME]) / DAY)
        for i in range(days):
            action_index = (self.agent_thread, 1, i)
            input_dict = populate_inputs(full_inputs=full_inputs,
                                         value=action_index,
                                         agent=True,
                                         agent_byr=True)
            log = Action(input_dict=input_dict,
                         days=i,
                         con=0,
                         thread_id=self.agent_thread,
                         turn=1)
            self.actions.push_action(action=log)
        return days

    def record_buyer_first_turn(self, full_inputs=None, days=None, agent_turns=None):
        con = agent_turns[1].agent_con()
        first_turn_index = (self.agent_thread, 1, days)
        input_dict = populate_inputs(full_inputs=full_inputs,
                                     value=first_turn_index,
                                     agent=True,
                                     agent_byr=True)
        first_offer = Action(input_dict=input_dict,
                             months=days,
                             con=con,
                             thread_id=self.translate_thread(self.agent_thread),
                             turn=1)
        self.actions.push_action(action=first_offer)
        # add remaining turns
        del agent_turns[1]

    def record_agent_thread(self, turns=None, thread_id=None, full_inputs=None):
        for t, turn in turns.items():
            time = turn.agent_time()
            days = (time - self.lookup[START_TIME]) / DAY
            index = (thread_id, t, 0)
            if turn.is_censored:
                input_dict = None
            else:
                input_dict = populate_inputs(full_inputs=full_inputs,
                                             value=index,
                                             agent=True,
                                             agent_byr=True)
            thread_id = self.translator.translate_thread(self.agent_thread)
            action = Action(con=turn.agent_con(),
                            censored=turn.is_censored,
                            days=days,
                            input_dict=input_dict,
                            thread_id=thread_id,
                            turn=t)
            self.actions.push_action(action=action)

    def _push_actions(self, params=None):
        full_inputs = params['inputs'][self.actions.model_name]
        days = self.record_agent_arrivals(full_inputs=full_inputs)
        # add first turn
        agent_turns = self.threads[self.agent_thread].get_agent_turns()
        self.record_buyer_first_turn(full_inputs=full_inputs,
                                     days=days,
                                     agent_turns=agent_turns)
        # record remaining threads
        self.record_agent_thread(turns=agent_turns,
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

    def is_agent_arrival(self, thread_id=None):
        return thread_id == self.agent_thread

    def is_agent_thread(self, thread_id=None):
        return thread_id == self.agent_thread

    def get_con(self, thread_id=None, turn=None, input_dict=None, time=None):
        """
        :return: np.float
        """
        true_id = self.translator.translate_thread(thread_id)
        con = self.threads[true_id].get_con(turn=turn, time=time, input_dict=input_dict)
        return con

    def get_msg(self, thread_id=None, turn=None, input_dict=None, time=None):
        """
        :return: np.float
        """
        true_id = self.translator.translate_thread(thread_id)
        msg = self.threads[true_id].get_msg(turn=turn, time=time, input_dict=input_dict)
        if msg:
            return 1.0
        else:
            return 0.0

    def get_delay(self, thread_id=None, turn=None, input_dict=None, time=None):
        """
        :return: int
        """
        true_id = self.translator.translate_thread(thread_id)
        delay = self.threads[true_id].get_delay(turn=turn, time=time, input_dict=input_dict)
        if delay == MAX_DELAY_ARRIVAL:
            return self.lookup[START_TIME] + MAX_DELAY_ARRIVAL - time
        else:
            return delay

    def get_agent_arrival(self, time=None, thread_id=None):
        arrival_time = self.translator.get_agent_arrival(check_time=time, thread_id=thread_id)
        return arrival_time

    def get_inter_arrival(self, thread_id=None, input_dict=None, time=None):
        if time == self.lookup[START_TIME]:
            assert input_dict is None
        else:
            assert input_dict is not None
        true_id = self.translator.translate_arrival(thread_id)
        if self.verbose:
            print('true id: {}'.format(true_id))
            print('actual id: {}'.format(thread_id))
        return self.arrivals[true_id].get_inter_arrival(check_time=time,
                                                        input_dict=input_dict)

    def get_hist(self, thread_id=None, input_dict=None, time=None):
        true_id = self.translator.translate_thread(thread_id)
        return self.arrivals[true_id].get_hist(check_time=time,
                                               input_dict=input_dict)

    def translate_thread(self, thread_id=None):
        return self.translator.translate_thread(thread_id)

    @staticmethod
    def check_bin(params=None, thread_id=None):
        thread1 = params[X_OFFER].xs(thread_id, level=THREAD)
        if len(thread1.index) == 1:
            assert thread1.loc[1, CON] == 1
            return True
        elif len(thread1.index) == 2:
            return thread1.loc[2, AUTO] and (thread1.loc[2, CON] == 1)

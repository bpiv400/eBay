"""
Environment for training the buyer agent
"""
import numpy as np
from constants import HOUR, DAY
from utils import get_months_since_lstg
from rlenv.env_consts import DELAY_EVENT, RL_ARRIVAL_EVENT
from rlenv.sources import RlSources
from rlenv.environments.AgentEnvironment import AgentEnvironment
from rlenv.events.Arrival import Arrival
from rlenv.events.Thread import RlThread


class BuyerEnvironment(AgentEnvironment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def turn_from_action(self, action=None):  # might not make sense
        return self.con_set[action]

    def get_reward(self):
        pass

    @property
    def horizon(self):
        return 100

    def _hours_since_lstg(self, priority):
        return int((priority - self.start_time) / HOUR)

    def get_arrival_time(self, event):
        """
        Gets an arrival time for the RL, forces the arrival to occur before
        the lstg ends
        :param event: RL_ARRIVAL_EVENT where the buyer has selected
        to make an offer
        :return: int giving arrival time
        """
        curr_interval = self._hours_since_lstg(event.priority)
        last_interval = curr_interval + 24
        input_dict = self.get_arrival_input_dict(event=event, first=True)
        seconds = self.get_arrival(input_dict=input_dict, first=True, time=event.priority,
                                   intervals=(curr_interval, last_interval))
        return seconds + event.priority

    def get_offer_time(self, priority):
        # query with delay model
        backstop = min(priority + self.end_time, priority + DELAY_INTERVAL)
        delay = int(np.random.uniform() * (backstop - priority))
        return max(delay, 1) + priority

    def process_rl_offer(self, event):

        # check whether the lstg expired, censoring this offer
        if self.is_lstg_expired(event):
            return self.process_lstg_expiration(event)
        if event.thread_expired():
            raise RuntimeError("Thread should never expire before agent offer")
        time_feats = self.time_feats.get_feats(thread_id=event.thread_id,
                                               time=event.priority)
        months_since_lstg = None
        if event.turn == 1:
            months_since_lstg = get_months_since_lstg(lstg_start=self.start_time,
                                                      time=event.priority)
        event.init_offer(months_since_lstg=months_since_lstg, time_feats=time_feats)
        offer = event.execute_offer()
        # update con outcomes
        # process post offer

    def process_offer(self, event):
        if isinstance(event, RlThread) and event.turn % 2 != 0:
            return self.process_rl_offer(event)
        else:
            return super().process_offer(event)

    def step(self, action):
        """
        Takes in a buyer action, updates the relevant event, then continues
        the simulation
        """
        con = self.turn_from_action(action=action)
        if self.last_event.event_type == RL_ARRIVAL_EVENT:
            if con == 0:
                # push rl arrival back into queue
                self.last_event.priority += DAY
                self.queue.push(self.last_event)
                self.last_event = None
                return self.run()
            else:
                sources = self.last_event.sources
                arrival_time = self.get_arrival_time(self.last_event.priority)
                sources.init_thread(hist=self.composer.hist)
                thread = RlThread(priority=arrival_time, sources=sources, con=con, rl_buyer=True,
                                  thread_id=self.thread_counter)
                self.thread_counter += 1
                self.queue.push(thread)
                self.last_event = None
                return self.run()
        else:
            if con == 0:
                return self.agent_tuple(done=True)
            else:
                offer_time = self.get_offer_time(self.last_event.priority)
                self.last_event.prep_rl_offer(con=con, priority=offer_time)
                self.queue.push(self.last_event)
                self.last_event = None
                return self.run()

    def reset(self):
        """
        Resets the environment by drawing a new listing,
        resetting the queue, adding a buyer rl arrival event, then
        running the environment
        :return: observation associated with the first rl arrival
        """
        self.reset_lstg()
        super().reset()
        rl_sources = RlSources(x_lstg=self.x_lstg)
        event = Arrival(priority=self.start_time, sources=rl_sources)
        self.queue.push(event)
        # should deterministically return RL_ARRIVAL_EVENT at start of lstg
        # lstg should never be complete at this point
        event, _ = super().run()
        event.update_arrival()
        self.last_event = event
        return self.get_obs()

    def is_agent_turn(self, event):
        """
        Indicates the buyer agent needs to take a turn when there's an
        agent arrival event or when it's buyer's turn in an RL thread
        :return:
        """
        if event.type == RL_ARRIVAL_EVENT:
            return not (self.is_lstg_expired(event))
        else:
            return (event.type == DELAY_EVENT and
                    event.turn % 2 == 1 and
                    isinstance(event, RlThread))

    def define_action_space(self, con_set=None):
        return
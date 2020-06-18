"""
Environment for training the buyer agent
"""
from collections import namedtuple
from constants import HOUR, DAY
from featnames import START_PRICE
from rlenv.const import DELAY_EVENT, RL_ARRIVAL_EVENT
from rlenv.sources import RlBuyerSources
from rlenv.environments.AgentEnvironment import AgentEnvironment
from rlenv.events.Arrival import Arrival
from rlenv.events.Thread import RlThread


class BuyerEnvironment(AgentEnvironment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # boolean indicating the sale that occurred took place in
        # an RL thread
        self.rl_event = None
        self.rl_sale = False

    def get_reward(self):
        """
        Returns the buyer reward for the current turn. For now, assumes
        that the buyer values the item at the start price
        """
        if self.rl_event.type == RL_ARRIVAL_EVENT:
            return 0.0
        elif not self.rl_event.is_sale():
            return 0.0
        else:
            return self.lookup[START_PRICE] - self.outcome.price

    def define_observation_class(self):
        return namedtuple('BuyerObs', self.composer.groupings)

    @property
    def horizon(self):
        return 34

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
            # rejection indicates delay without action
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
                thread = RlThread(priority=arrival_time, sources=sources,
                                  con=con, rl_buyer=True,
                                  thread_id=self.thread_counter)
                self.rl_event = thread
                self.thread_counter += 1
                self.queue.push(thread)
                self.last_event = None
                return self.run()
        else:
            # expiration rejection ends the trajectory
            if con == 0:
                return self.agent_tuple(done=True)
            else:
                # otherwise sample offer time
                offer_time = self.get_offer_time(self.last_event)
                self.last_event.prep_rl_offer(con=con, priority=offer_time)
                self.queue.push(self.last_event)
                self.last_event = None
                return self.run()

    def run(self):
        event, lstg_complete = super().run()
        # set last event flag to point to rl event
        self.last_event = self.rl_event
        if event is not self.rl_event and not lstg_complete:
            raise RuntimeError("Other threads should only return "
                               "to agent when the lstg ends")
        return self.agent_tuple(done=lstg_complete)

    def reset(self):
        """
        Resets the environment by drawing a new listing,
        resetting the queue, adding a buyer rl arrival event, then
        running the environment
        :return: observation associated with the first rl arrival
        """
        self.reset_lstg()
        super().reset()
        rl_sources = RlBuyerSources(x_lstg=self.x_lstg)
        event = Arrival(priority=self.start_time, sources=rl_sources,
                        rl=True)
        self.rl_event = event
        self.queue.push(event)
        # should deterministically return RL_ARRIVAL_EVENT at start of lstg
        # lstg should never be complete at this point
        event, _ = super().run()
        if event.type != RL_ARRIVAL_EVENT:
            raise RuntimeError("Bad assumption about first event")
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

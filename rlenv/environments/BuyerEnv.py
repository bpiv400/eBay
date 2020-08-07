import numpy as np
from rlpyt.utils.collections import namedarraytuple
from constants import HOUR, DAY, POLICY_BYR, PCTILE_DIR
from featnames import START_PRICE, BYR_HIST
from utils import load_sizes, unpickle
from rlenv.const import DELAY_EVENT, RL_ARRIVAL_EVENT
from rlenv.Sources import RlBuyerSources
from rlenv.environments.AgentEnv import AgentEnv
from rlenv.events.Arrival import RlArrival
from rlenv.events.Thread import RlThread

BuyerObs = namedarraytuple('BuyerObs',
                           list(load_sizes(POLICY_BYR)['x'].keys()))


class BuyerEnv(AgentEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.item_value = None
        self.agent_thread = None  # thread number for agent buyer

        # for drawing experience
        path = PCTILE_DIR + '{}.pkl'.format(BYR_HIST)
        self.hist_pctile = unpickle(path).values

    def reset(self, next_lstg=True):
        """
        Resets the environment by drawing a new listing,
        resetting the queue, adding a buyer rl arrival event, then
        running the environment
        :return: observation associated with the first rl arrival
        """
        self.init_reset(next_lstg=next_lstg)  # in AgentEnvironment
        self.item_value = self.lookup[START_PRICE]  # TODO: allow for different values

        # put rl arrival in queue
        rl_sources = RlBuyerSources(x_lstg=self.x_lstg,
                                    hist=self._draw_hist())
        event = RlArrival(priority=self.start_time, sources=rl_sources)
        self.queue.push(event)
        event_copy = event

        # should return same rl arrival event
        event, lstg_complete = super().run()
        assert not lstg_complete
        assert event is event_copy
        self.last_event = event

        # return observation to agent
        return self.get_obs(event=event, done=False)

    def step(self, action):
        """
        Takes in a buyer action, updates the relevant event, then continues
        the simulation
        """
        con = self.turn_from_action(action=action)
        if self.last_event.type == RL_ARRIVAL_EVENT:
            return self._step_arrival(con=con)
        else:
            return self._step_thread(con=con)

    def _step_arrival(self, con=None):
        # rejection indicates delay without action
        if con == 0:
            self.last_event.delay()  # adds DAY to priority, updates sources
            self.queue.push(self.last_event)

        else:
            sources = self.last_event.sources
            arrival_time = self._get_arrival_time(self.last_event)
            thread = RlThread(priority=arrival_time,
                              sources=sources,
                              con=con,
                              rl_buyer=True)
            self.queue.push(thread)

        return self.run()

    def _step_thread(self, con=None):
        offer_time = self.get_offer_time(self.last_event)
        self.last_event.prep_rl_offer(con=con, priority=offer_time)
        self.queue.push(self.last_event)
        return self.run()

    def run(self):
        event, lstg_complete = super().run()
        return self.agent_tuple(done=lstg_complete, event=event)

    def get_reward(self):
        """
        Returns the buyer reward for the current turn. For now, assumes
        that the buyer values the item at the start price
        """
        # no sale
        if self.outcome is None or not self.outcome.sale:
            return 0.

        # sale to different buyer
        if self.outcome.thread != self.agent_thread:
            return 0.

        # sale to agent buyer
        return self.item_value - self.outcome.price

    def is_agent_turn(self, event):
        """
        Indicates the buyer agent needs to take a turn when there's an
        agent arrival event or when it's buyer's turn in an RL thread
        :return bool: True if agent buyer takes an action
        """
        if event.type == RL_ARRIVAL_EVENT:
            return not (self.is_lstg_expired(event))
        elif event.type == DELAY_EVENT:
            return event.turn % 2 == 1 and isinstance(event, RlThread)
        else:
            return False

    def process_offer(self, event):
        if isinstance(event, RlThread) and event.turn % 2 != 0:
            return self.process_rl_offer(event)
        else:
            return super().process_offer(event)

    def _hours_since_lstg(self, priority):
        return int((priority - self.start_time) / HOUR)

    def _get_arrival_time(self, event):
        """
        Gets an arrival time for the RL, forces the arrival to occur before
        the lstg ends
        :param event: RL_ARRIVAL_EVENT where the buyer has selected
        to make an offer
        :return: int giving arrival time
        """
        curr = self._hours_since_lstg(event.priority)
        seconds = self.get_first_arrival(time=event.priority,
                                         thread_id=self.thread_counter,
                                         intervals=(curr, curr + 24))
        return seconds + event.priority

    def _draw_hist(self):
        q = np.random.uniform()
        idx = np.searchsorted(self.hist_pctile, q) - 1
        hist = self.hist_pctile[idx]
        return hist

    def _get_max_return(self):
        return self.item_value

    @property
    def horizon(self):
        return 34

    @property
    def _obs_class(self):
        return BuyerObs

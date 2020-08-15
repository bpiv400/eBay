import numpy as np
from rlpyt.utils.collections import namedarraytuple
from constants import HOUR, POLICY_BYR, PCTILE_DIR
from featnames import START_PRICE, BYR_HIST
from utils import load_sizes, unpickle, get_months_since_lstg
from rlenv.const import DELAY_EVENT, RL_ARRIVAL_EVENT, CLOCK_MAP
from rlenv.Sources import RlBuyerSources
from rlenv.environments.AgentEnv import AgentEnv
from rlenv.events.Arrival import RlArrival
from rlenv.events.Thread import RlThread

BuyerInfo = namedarraytuple("BuyerInfo",
                            ["months",
                             "max_return",
                             "num_delays",
                             "num_offers"])
BuyerObs = namedarraytuple('BuyerObs',
                           list(load_sizes(POLICY_BYR)['x'].keys()))


class BuyerEnv(AgentEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.item_value = None
        self.agent_thread = None
        self.num_delays = None

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
        self.agent_thread = None
        self.num_delays = 0

        # put rl arrival in queue
        rl_sources = RlBuyerSources(x_lstg=self.x_lstg, hist=self._draw_hist())
        event = RlArrival(priority=self.start_time, sources=rl_sources)
        self.queue.push(event)

        event, lstg_complete = super().run()  # returns same RlArrival
        assert not lstg_complete
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
            if self.last_event.priority >= self.end_time:
                return self.agent_tuple(done=True, event=self.last_event)
            self.queue.push(self.last_event)
            self.num_delays += 1

        else:
            sources = self.last_event.sources
            arrival_time = self._get_arrival_time(self.last_event)
            thread = RlThread(priority=arrival_time,
                              sources=sources,
                              con=con,
                              rl_buyer=True)
            self.queue.push(thread)

        self.last_event = None
        return self.run()

    def _step_thread(self, con=None):
        assert self.last_event.turn in [3, 5, 7]
        # trajectory ends with buyer reject
        if con == 0. or (con < 1 and self.last_event.turn == 7):
            return self.agent_tuple(done=True, event=self.last_event)

        # otherwise, draw an offer time and put event in queue
        delay_seconds = self.draw_agent_delay(self.last_event)
        offer_time = delay_seconds + self.last_event.priority
        self.last_event.prep_rl_offer(con=con, priority=offer_time)
        self.queue.push(self.last_event)
        self.last_event = None
        return self.run()

    def run(self):
        event, lstg_complete = super().run()
        if not lstg_complete:
            self.last_event = event
        return self.agent_tuple(done=lstg_complete, event=event)

    def is_agent_turn(self, event):
        """
        Indicates the buyer agent needs to take a turn when there's an
        agent arrival event or when it's buyer's turn in an RL thread
        :return bool: True if agent buyer takes an action
        """
        if event.type == RL_ARRIVAL_EVENT:
            return not (self.is_lstg_expired(event))
        if event.type == DELAY_EVENT:
            return event.turn % 2 == 1 and isinstance(event, RlThread)
        return False

    def process_offer(self, event):
        if isinstance(event, RlThread) and event.turn % 2 == 1:
            if event.turn == 1:  # housekeeping for buyer's first turn
                self.last_arrival_time = event.priority
                event.set_thread_id(self.thread_counter)
                self.thread_counter += 1
                self.agent_thread = event.thread_id

            # check whether the lstg expired, censoring this offer
            if self.is_lstg_expired(event):
                return self.process_lstg_expiration(event)

            if event.thread_expired():
                raise RuntimeError("Thread should not expire before byr agent offer")

            # initalize offer
            time_feats = self.time_feats.get_feats(thread_id=event.thread_id,
                                                   time=event.priority)
            months_since_lstg = None
            if event.turn == 1:
                months_since_lstg = get_months_since_lstg(lstg_start=self.start_time,
                                                          time=event.priority)
            event.init_rl_offer(months_since_lstg=months_since_lstg,
                                time_feats=time_feats)

            # execute offer
            offer = event.execute_offer()
            self.num_offers += 1

            # return True if lstg is over
            return self.process_post_offer(event, offer)
        else:
            return super().process_offer(event)

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

    def get_obs(self, event=None, done=None):
        if event.sources() is None or event.turn is None:
            raise RuntimeError("Missing arguments to get observation")
        if CLOCK_MAP in event.sources() and BYR_HIST in event.sources():
            obs_dict = self.composer.build_input_dict(model_name=None,
                                                      sources=event.sources(),
                                                      turn=event.turn)
        else:  # incomplete sources; triggers warning in AgentModel
            assert done
            obs_dict = self.empty_obs_dict
        return self._obs_class(**obs_dict)

    def get_info(self, event=None):
        return BuyerInfo(months=self._get_months(event.priority),
                         max_return=self.item_value,
                         num_delays=self.num_delays,
                         num_offers=self.num_offers)

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

    @property
    def horizon(self):
        return 34

    @property
    def _obs_class(self):
        return BuyerObs

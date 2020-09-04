import numpy as np
from rlpyt.utils.collections import namedarraytuple
from constants import HOUR, POLICY_BYR, PCTILE_DIR, INTERVAL_CT_ARRIVAL, \
    DAY, MAX_DAYS
from featnames import START_PRICE, BYR_HIST
from rlenv.util import get_clock_feats, get_con_outcomes
from utils import load_sizes, unpickle, get_days_since_lstg
from rlenv.const import FIRST_OFFER, DELAY_EVENT, OFFER_EVENT, RL_ARRIVAL_EVENT
from rlenv.Sources import ThreadSources
from agent.envs.AgentEnv import AgentEnv
from agent.util import define_con_set
from rlenv.events.Event import Event
from rlenv.events.Thread import Thread

BuyerObs = namedarraytuple('BuyerObs',
                           list(load_sizes(POLICY_BYR)['x'].keys()))


class BuyerEnv(AgentEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent_thread = None
        self.hist = None

        # for drawing experience
        path = PCTILE_DIR + '{}.pkl'.format(BYR_HIST)
        self.hist_pctile = unpickle(path).values

    def reset(self, next_lstg=True, hist=None):
        """
        Resets the environment by drawing a new listing,
        resetting the queue, adding a buyer rl arrival event, then
        running the environment
        :return: observation associated with the first rl arrival
        """
        self.init_reset(next_lstg=next_lstg)  # in BuyerEnv
        if hist is None:
            self.hist = self._draw_hist()
        else:  # for TestGenerator
            self.hist = hist

        while True:
            # put an RL arrival into the queue and run environment
            self.queue.push(self._create_rl_event(self.start_time))
            event, lstg_complete = super().run()  # calls EBayEnv.run()

            # when rl buyer arrives, create a prospective thread but do not put in queue
            if not lstg_complete:
                thread = self._create_thread(event.priority)
                self.last_event = thread
                return self.get_obs(event=thread, done=False)

            # item sells before rl buyer arrival
            elif next_lstg:
                self.init_reset(next_lstg=True)
            else:  # for EvalGenerator
                return None

    def init_reset(self, next_lstg=None):
        super().init_reset(next_lstg=next_lstg)
        self.item_value = self.lookup[START_PRICE]  # TODO: allow for different values
        self.agent_thread = None

    def step(self, action):
        """
        Takes in a buyer action, updates the relevant event, then continues
        the simulation
        """
        con = self.turn_from_action(action=action)
        event_type = self.last_event.type
        if event_type == FIRST_OFFER:
            if con == 0:
                return self._step_arrival_delay()
            else:
                return self._step_first_offer(con)
        elif event_type == OFFER_EVENT:
            return self._step_thread(con)
        else:
            raise ValueError('Invalid event type: {}'.format(event_type))

    def _step_arrival_delay(self):
        next_midnight = (self.last_event.priority // DAY + 1) * DAY
        if next_midnight == self.end_time:
            return self.agent_tuple(done=True, event=self.last_event)
        self.queue.push(self._create_rl_event(next_midnight))
        self.num_delays += 1
        self.last_event = None
        return self.run()

    def _step_first_offer(self, con=None):
        # copy event
        thread = self.last_event
        self.last_event = None

        # save thread id and increment counter
        self.agent_thread = thread.thread_id
        self.thread_counter += 1

        # record and print
        hist = thread.sources()[BYR_HIST]
        self.record(thread, byr_hist=hist, agent=True)
        if self.verbose:
            print('Agent thread {} initiated | Buyer hist: {}'.format(
                self.agent_thread, hist))

        # create and execute offer
        return self._execute_offer(con=con, thread=thread)

    def _step_thread(self, con=None):
        # copy event
        thread = self.last_event
        self.last_event = None

        # error checking
        assert thread.turn in [3, 5, 7]
        assert not thread.thread_expired()

        # trajectory ends with buyer reject
        if con == 0. or (con < 1 and thread.turn == 7):
            return self.agent_tuple(done=True, event=thread)

        # otherwise, execute the offer
        return self._execute_offer(con=con, thread=thread)

    def _execute_offer(self, con=None, thread=None):
        self.num_offers += 1
        con_outcomes = get_con_outcomes(con=con,
                                        sources=thread.sources(),
                                        turn=thread.turn)
        offer = thread.update_con_outcomes(con_outcomes=con_outcomes)
        lstg_complete = self.process_post_offer(thread, offer)
        if lstg_complete:
            return self.agent_tuple(event=thread, done=lstg_complete)
        return self.run()

    def is_agent_turn(self, event):
        """
        When the agents thread comes back on the buyer's turn, it's time to act.
        :return bool: True if agents buyer takes an action
        """
        if event.turn % 2 == 1:
            if event.type == RL_ARRIVAL_EVENT:
                return True
            if isinstance(event, Thread) and event.agent:
                return not self.is_lstg_expired(event) and not event.thread_expired()
        return False

    def run(self):
        while True:
            event, lstg_complete = super().run()  # calls EBayEnv.run()

            # for RL buyer delay, draw delay and put back in queue
            if event.type == DELAY_EVENT:  # sample delay and put back in queue
                delay_seconds = self.draw_agent_delay(event)
                event.update_delay(seconds=delay_seconds)
                self.queue.push(event)

            else:
                # agent's turn to make a concession
                if not lstg_complete:

                    # replace RL arrival with thread
                    if event.type == RL_ARRIVAL_EVENT:
                        event = self._create_thread(event.priority)

                    self.prepare_offer(event)
                    self.last_event = event  # save event for step methods

                # return if done or time for agent action
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

        # sale to agents buyer
        if self.verbose:
            print('Sale to RL buyer. List price: {}. Paid price: {}'.format(
                self.item_value, self.outcome.price))
        assert self.item_value >= self.outcome.price
        return self.item_value - self.outcome.price

    def _create_rl_event(self, midnight=None):
        assert midnight % DAY == 0
        arrival_time = self._get_arrival_time(midnight)
        return Event(event_type=RL_ARRIVAL_EVENT, priority=arrival_time)

    def _create_thread(self, arrival_time=None):
        # construct sources
        sources = ThreadSources(x_lstg=self.x_lstg)
        clock_feats = get_clock_feats(arrival_time)
        time_feats = self.time_feats.get_feats(time=arrival_time,
                                               thread_id=self.thread_counter)
        days_since_lstg = get_days_since_lstg(lstg_start=self.start_time,
                                              time=arrival_time)
        sources.prepare_hist(clock_feats=clock_feats,
                             time_feats=time_feats,
                             days_since_lstg=days_since_lstg)

        # construct thread, then return
        thread = Thread(priority=arrival_time, agent=True)
        thread.set_id(self.thread_counter)
        thread.init_thread(sources=sources, hist=self.hist)

        return thread

    def _get_arrival_time(self, priority):
        """
        Gets an arrival time for the RL, forces the arrival to occur before
        the lstg ends
        :param int priority: midnight of some day in listing window
        :return: int giving arrival time
        """
        curr = int((priority - self.start_time) / HOUR)
        intervals_in_day = int(INTERVAL_CT_ARRIVAL / MAX_DAYS)
        intervals = (curr, curr + intervals_in_day)
        seconds = self.get_first_arrival(intervals=intervals)
        return priority + seconds

    def _draw_hist(self):
        q = np.random.uniform()
        idx = np.searchsorted(self.hist_pctile, q) - 1
        hist = self.hist_pctile[idx]
        return hist

    def _define_con_set(self, con_set):
        return define_con_set(con_set=con_set, byr=True)

    @property
    def horizon(self):
        return MAX_DAYS + 3

    @property
    def _obs_class(self):
        return BuyerObs

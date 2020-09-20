import numpy as np
from rlpyt.utils.collections import namedarraytuple
from rlenv.util import get_clock_feats, get_con_outcomes
from utils import load_sizes, unpickle, get_days_since_lstg
from rlenv.const import FIRST_OFFER, DELAY_EVENT, OFFER_EVENT, RL_ARRIVAL_EVENT
from rlenv.Sources import ThreadSources
from agent.envs.AgentEnv import AgentEnv, EventLog
from agent.util import define_con_set, load_values
from rlenv.events.Event import Event
from rlenv.events.Thread import Thread
from agent.const import LSTG_SIM_CT
from constants import HOUR, POLICY_BYR, PCTILE_DIR, INTERVAL_CT_ARRIVAL, \
    DAY, MAX_DAYS, RL_BYR
from featnames import BYR_HIST, START_PRICE

BuyerObs = namedarraytuple('BuyerObs',
                           list(load_sizes(POLICY_BYR)['x'].keys()))


class BuyerEnv(AgentEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hist = None
        self.agent_thread = None
        self.lstg_sim_ct = None

        # for drawing experience
        path = PCTILE_DIR + '{}.pkl'.format(BYR_HIST)
        self.hist_pctile = unpickle(path).values

        # values
        if not self.test:
            self.values = load_values(part=RL_BYR)

    def reset(self, hist=None):
        """
        Resets the environment by drawing a new listing,
        resetting the queue, adding a buyer rl arrival event, then
        running the environment
        :return: observation associated with the first rl arrival
        """
        self.init_reset(hist=hist)  # in ByrEnv
        while True:
            # put an RL arrival into the queue and run environment
            self.queue.push(self._create_rl_event(self.start_time))
            event, lstg_complete = super().run()  # calls EBayEnv.run()

            # when rl buyer arrives, create a prospective thread but do not put in queue
            if not lstg_complete:
                thread = self._create_thread(event.priority)
                self.curr_event = thread
                return self.get_obs(event=thread, done=False)

            # item sells before rl buyer arrival
            elif not self.test:  # for training
                self.init_reset()
            else:  # for testing and evaluation
                return None

    def init_reset(self, hist=None, push_arrival=True):
        self.curr_event = None
        self.last_event = None
        self.num_actions = 0
        if not self.test:
            if not self.has_next_lstg():
                raise RuntimeError("Out of lstgs")
            if self.lstg_sim_ct is None or self.lstg_sim_ct == LSTG_SIM_CT:
                self.next_lstg()
                self.lstg_sim_ct = 1
            else:
                self.lstg_sim_ct += 1
        super().reset(push_arrival)  # calls EBayEnv.reset()
        self.agent_thread = 0
        if hist is None:
            self.hist = self._draw_hist()
        else:  # for TestGenerator
            self.hist = hist

    def step(self, action):
        """
        Takes in a buyer action, updates the relevant event, then continues
        the simulation
        """
        self.last_event = EventLog(priority=self.curr_event.priority,
                                   thread_id=self.curr_event.thread_id,
                                   turn=self.curr_event.turn)
        con = self.turn_from_action(action=action)
        event_type = self.curr_event.type
        self.num_actions += 1
        if event_type == FIRST_OFFER:
            if con == 0:
                return self._step_arrival_delay()
            else:
                return self._step_first_offer(con)
        elif event_type == OFFER_EVENT:
            assert self.curr_event.turn in [3, 5, 7]
            assert not self.curr_event.thread_expired()
            return self._execute_offer(con=con)
        else:
            raise ValueError('Invalid event type: {}'.format(event_type))

    def _step_arrival_delay(self):
        if self.recorder is not None:
            self.recorder.add_agent_delay(self.curr_event.priority)
        next_midnight = (self.curr_event.priority // DAY + 1) * DAY
        if next_midnight == self.end_time:
            return self.agent_tuple(done=True,
                                    event=self.curr_event,
                                    info=self.get_info(self.last_event))
        self.queue.push(self._create_rl_event(next_midnight))
        self.curr_event = None
        return self.run()

    def _step_first_offer(self, con=None):
        # save thread id and increment counter
        self.agent_thread = self.curr_event.thread_id
        self.thread_counter += 1

        # record and print
        hist = self.curr_event.sources()[BYR_HIST]
        self.record(self.curr_event, byr_hist=hist, agent=True)
        if self.verbose:
            print('Agent thread {} initiated | Buyer hist: {}'.format(
                self.agent_thread, hist))

        # create and execute offer
        return self._execute_offer(con=con)

    def _execute_offer(self, con=None):
        con_outcomes = get_con_outcomes(con=con,
                                        sources=self.curr_event.sources(),
                                        turn=self.curr_event.turn)
        offer = self.curr_event.update_con_outcomes(con_outcomes=con_outcomes)
        lstg_complete = self.process_post_offer(self.curr_event, offer)
        if lstg_complete or con == 0:
            return self.agent_tuple(event=self.curr_event,
                                    done=True,
                                    info=self.get_info(self.last_event))
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
                    self.curr_event = event  # save event for step methods

                # return if done or time for agent action
                return self.agent_tuple(done=lstg_complete,
                                        event=event,
                                        info=self.get_info(self.last_event))

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
        # value = self.values.loc[self.loader.lstg]
        value = self.lookup[START_PRICE]
        if self.verbose:
            print('Sale to RL buyer. Price: {0:.2f}. Value: {1:.2f}'.format(
                self.outcome.price, value))
        return value - self.outcome.price

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

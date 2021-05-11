import numpy as np
from rlpyt.utils.collections import namedarraytuple
from rlenv.util import get_con_outcomes
from utils import load_sizes
from rlenv.const import FIRST_OFFER, DELAY_EVENT, OFFER_EVENT, ARRIVAL
from agent.envs.AgentEnv import AgentEnv, EventLog
from constants import BYR
from featnames import START_PRICE

BuyerObs = namedarraytuple('BuyerObs', list(load_sizes(BYR)['x'].keys()))


class BuyerEnv(AgentEnv):
    def __init__(self, **kwargs):
        super().__init__(byr=True, **kwargs)
        self.arrivals_first = True
        self.agent_thread = None

    def is_agent_turn(self, event):
        """
        When the agents thread comes back on the buyer's turn, it's time to act.
        :return bool: True if agents buyer takes an action
        """
        if event.type == ARRIVAL:
            return False

        # catch first buyer, determine which buyer is agent
        if event.thread_id == 1 and event.type == FIRST_OFFER:
            assert self.agent_thread is None
            self.agent_thread = np.random.randint(1, self.thread_counter + 1)
            if self.verbose:
                print('{} buyer{}, agent is #{}.'.format(
                    self.thread_counter,
                    's' if self.thread_counter > 1 else '',
                    self.agent_thread)
                )
            return self.agent_thread == 1

        # catch buyer turns on agent thread
        assert self.agent_thread is not None
        if event.thread_id == self.agent_thread and event.turn % 2 == 1:
            return not self.is_lstg_expired(event) and not event.thread_expired()

        # other buyers or seller turn
        return False

    def init_reset(self):
        super().init_reset()
        self.agent_thread = None

    def reset(self):
        """
        Resets the environment by drawing a new listing,
        resetting the queue, adding a buyer rl arrival event, then
        running the environment
        :return: observation associated with the first rl arrival
        """
        self.init_reset()  # in BuyerEnv
        while True:
            event, lstg_complete = super().run()  # calls EBayEnv.run()

            # when agent buyer arrives, create a thread and get an offer
            if not lstg_complete:
                if event.thread_id == self.agent_thread:
                    if event.type != FIRST_OFFER:
                        raise ValueError('Incorrect event type: {}'.format(event.type))
                    self.create_thread(event, is_agent=True)
                    self.curr_event = event
                    return self.get_obs(event=self.curr_event, done=False)

            # listing ends before RL arrival
            elif self.train:  # queue up next listing
                self.init_reset()
            else:  # for testing and evaluation
                return None

    def step(self, action):
        """
        Takes in a buyer action, updates the relevant event, then continues
        the simulation
        """
        con = self.turn_from_action(turn=self.curr_event.turn,
                                    action=action)
        if 0. < con < 1.:
            self.num_actions += 1

        self.last_event = EventLog(priority=self.curr_event.priority,
                                   thread_id=self.curr_event.thread_id,
                                   turn=self.curr_event.turn)

        event_type = self.curr_event.type
        if event_type == FIRST_OFFER and con == 0:
            return self._step_arrival_delay()
        elif event_type == OFFER_EVENT or (event_type == FIRST_OFFER and con > 0):
            return self._execute_offer(con=con)
        else:
            raise ValueError('Invalid event type: {}'.format(event_type))

    def _step_arrival_delay(self):
        # record offer
        if self.recorder is not None:
            self.recorder.add_buyer_walk(
                event=self.curr_event,
                time_feats=self.curr_event.sources.offer_prev_time
            )

        # print
        if self.verbose:
            print('Agent buyer walks at {}.'.format(self.curr_event.priority))
        return self.agent_tuple(done=True,
                                event=self.curr_event,
                                last_event=self.last_event)

    def _execute_offer(self, con=None):
        assert not self.curr_event.thread_expired()
        con_outcomes = get_con_outcomes(con=con,
                                        sources=self.curr_event.sources(),
                                        turn=self.curr_event.turn)
        offer = self.curr_event.update_con_outcomes(con_outcomes=con_outcomes)
        lstg_complete = self.process_post_offer(self.curr_event, offer)
        if lstg_complete or con == 0:
            return self.agent_tuple(event=self.curr_event,
                                    done=True,
                                    last_event=self.last_event)
        return self.run()

    def run(self):
        while True:
            event, lstg_complete = super().run()  # calls EBayEnv.run()

            # for RL buyer delay, draw delay and put back in queue
            if event.type == DELAY_EVENT:  # sample delay and put back in queue
                delay_seconds = self.draw_agent_delay(event)
                event.update_delay(seconds=delay_seconds)
                self.queue.push(event)

            # agent's turn to make a concession
            else:
                if not lstg_complete:
                    assert event.type == OFFER_EVENT
                    self.prepare_offer(event)
                    self.curr_event = event  # save event for step methods

                # return if done or time for agent action
                return self.agent_tuple(done=lstg_complete,
                                        event=event,
                                        last_event=self.last_event)

    def get_reward(self):
        """
        Returns the buyer reward for the current turn.
        """
        # turn cost penalty
        penalty = self.num_actions * self.turn_cost

        # no sale
        if self.outcome is None or not self.outcome.sale:
            return -penalty, False

        # sale to different buyer
        if self.outcome.thread != self.agent_thread:
            return -penalty, False

        # sale to agent buyer
        value = self.delta * self.lookup[START_PRICE]
        if self.verbose:
            print('Sale to RL buyer. Price: ${0:.2f}.'.format(
                self.outcome.price))

        return value - self.outcome.price - penalty, True

    @property
    def horizon(self):
        return 4

    @property
    def _obs_class(self):
        return BuyerObs

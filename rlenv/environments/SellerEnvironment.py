from collections import namedtuple
from constants import MAX_DELAY_TURN
from rlenv.environments.AgentEnvironment import AgentEnvironment
from rlenv.const import OFFER_EVENT, DELAY_EVENT
from rlenv.util import get_con_outcomes
from rlenv.events.Thread import RlThread


class SellerEnvironment(AgentEnvironment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def is_agent_turn(self, event):
        """
        Checks whether the agent should take a turn
        :param rlenv.events.Thread.Thread event:
        :return: bool
        """
        if self.delay:
            return self._delay_agent_turn(event)
        else:
            return self._con_agent_turn(event)

    @staticmethod
    def _delay_agent_turn(event):
        return event.type == DELAY_EVENT and event.turn % 2 == 0

    def _con_agent_turn(self, event):
        if event.type == OFFER_EVENT and event.turn % 2 == 0:
            return not (self.is_lstg_expired(event) or event.thread_expired())
        else:
            return False

    def run(self):
        # until EbayEnvironment.run() returns an agent action
        while True:
            event, lstg_complete = super().run()
            self.last_event = event
            # update most recent time/clock features if sampling an agent
            # concession from the con-only agent
            if not lstg_complete and not self.delay:
                self.prepare_offer(event)
            if not lstg_complete or self.outcome.sale:
                return self.agent_tuple(done=lstg_complete)
            else:
                self.relist()

    # restructure
    def reset(self):
        self.reset_lstg()
        super().reset()  # calls EBayEnvironment.reset()
        while True:
            event, lstg_complete = super().run()  # calls EBayEnvironment.run()
            # if the lstg isn't complete that means it's time to sample an agent action
            if not lstg_complete:
                self.last_event = event
                # if this isn't a delay agent, update sources with recent time/clock feats
                if not self.delay:
                    self.prepare_offer(event)
                return self.get_obs(sources=event.sources(), turn=event.turn)
            # if the lstg is complete
            else:
                # check whether it's expired -- if so, relist
                if not self.outcome.sale:
                    self.relist()
                # otherwise, there's been a buy it now sale w/o a seller action,
                # so we move onto the next lstg
                else:
                    self.reset_lstg()
                    super().reset()

    def relist(self):
        self.relist_count += 1
        super().reset()

    def step(self, action):
        """
        Process int giving concession/delay
        :param action: int returned from agent
        :return: tuple described in rlenv
        """
        con = self.turn_from_action(action)
        if self.delay:
            return self._delay_agent_step(con)
        else:
            return self._con_agent_step(con)

    def _con_agent_step(self, con):
        con_outcomes = get_con_outcomes(con=con,
                                        sources=self.last_event.sources(),
                                        turn=self.last_event.turn)
        offer = self.last_event.update_con_outcomes(con_outcomes=con_outcomes)
        lstg_complete = self.process_post_offer(self.last_event, offer)
        if lstg_complete:
            return self.agent_tuple(done=lstg_complete)
        self.last_event = None
        return self.run()

    def _delay_agent_step(self, con):
        # not expiration rejection
        if con <= 1:
            offer_time = self.get_offer_time(self.last_event)
            self.last_event.prep_rl_offer(con=con, priority=offer_time)
        # expiration rejection
        else:
            self.last_event.update_delay(seconds=MAX_DELAY_TURN)
        self.queue.push(self.last_event)
        self.last_event = None
        return self.run()

    def make_thread(self, priority):
        return RlThread(priority=priority, rl_buyer=False,
                        thread_id=self.thread_counter)

    def define_observation_class(self):
        return namedtuple("SellerObs", self.composer.groupings)

    def get_reward(self):
        if not self.last_event.is_sale():
            return 0.0
        else:
            return self.outcome.price * (1-self.cut)

    @property
    def horizon(self):
        return 100

from collections import namedtuple
from constants import MAX_DELAY_TURN
from utils import load_sizes
from agent.util import get_agent_name
from rlenv.const import DELAY_EVENT
from rlenv.environments.AgentEnvironment import AgentEnvironment
from rlenv.events.Thread import RlThread
from rlenv.Recorder import Recorder

# SELLER_GROUPINGS = list(load_sizes(get_agent_name(byr=False, delay=False, policy=True))['x'].keys())
# SellerObs = namedtuple("SellerObs", SELLER_GROUPINGS)
SELLER_DELAY_GROUPINGS = list(load_sizes(get_agent_name(byr=False))['x'].keys())
SellerDelayObs = namedtuple("SellerDelayObs", SELLER_DELAY_GROUPINGS)


class SellerEnvironment(AgentEnvironment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def is_agent_turn(self, event):
        """
        Checks whether the agent should take a turn
        :param rlenv.events.Thread.Thread event:
        :return: bool
        """
        return self._delay_agent_turn(event)

    @staticmethod
    def _delay_agent_turn(event):
        return event.type == DELAY_EVENT and event.turn % 2 == 0

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
        self.next_lstg()
        super().reset()  # calls EBayEnvironment.reset()
        while True:
            event, lstg_complete = super().run()  # calls EBayEnvironment.run()
            # if the lstg isn't complete that means it's time to sample an agent action
            if not lstg_complete:
                self.last_event = event
                return self.get_obs(sources=event.sources(), turn=event.turn)
            # if the lstg is complete
            else:
                # check whether it's expired -- if so, relist
                if not self.outcome.sale:
                    self.relist()
                # otherwise, there's been a buy it now sale w/o a seller action,
                # so we move onto the next lstg
                else:
                    self.next_lstg()
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
        return self._delay_agent_step(con)

    def _delay_agent_step(self, con):
        # not expiration rejection
        if con <= 1:
            offer_time = self.get_offer_time(self.last_event)
            delay = offer_time - self.last_event.priority
            self.last_event.prep_rl_offer(con=con, priority=offer_time)
        # expiration rejection
        else:
            delay = MAX_DELAY_TURN
            self.last_event.update_delay(seconds=MAX_DELAY_TURN)
        self.queue.push(self.last_event)
        self.last_event = None
        if self.verbose:
            Recorder.print_agent_turn(con=con,
                                      delay=delay / MAX_DELAY_TURN)
        return self.run()

    def process_offer(self, event):
        if event.turn % 2 == 0 and self.delay:
            return self.process_rl_offer(event)
        else:
            return super().process_offer(event)

    def make_thread(self, priority):
        return RlThread(priority=priority, rl_buyer=False,
                        thread_id=self.thread_counter)

    def define_observation_class(self):
        return SellerDelayObs

    def get_reward(self):
        if not self.last_event.is_sale():
            return 0.0
        else:
            return self.outcome.price * (1-self.cut)

    @property
    def horizon(self):
        return 100

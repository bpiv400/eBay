from rlpyt.utils.collections import namedarraytuple
from featnames import START_PRICE
from constants import MAX_DELAY_TURN, POLICY_SLR, LISTING_FEE
from utils import load_sizes
from rlenv.const import DELAY_EVENT
from rlenv.environments.AgentEnv import AgentEnv
from rlenv.events.Thread import RlThread
from rlenv.generate.Recorder import Recorder

SellerObs = namedarraytuple("SellerObs",
                            list(load_sizes(POLICY_SLR)['x'].keys()))


class SellerEnv(AgentEnv):
    """
    Abstract class for implementing seller agent environment.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, action):
        """
        Process int giving concession/delay
        :param action: int returned from agent
        :return: tuple described in rlenv
        """
        con = self.turn_from_action(action)

        # not expiration rejection
        if con <= 1:
            offer_time = self.get_offer_time(self.last_event)
            delay = (offer_time - self.last_event.priority) / MAX_DELAY_TURN
            self.last_event.prep_rl_offer(con=con, priority=offer_time)

        # expiration rejection
        else:
            delay = 1.
            self.last_event.update_delay(seconds=MAX_DELAY_TURN)

        # put event in queue
        self.queue.push(self.last_event)

        if self.verbose:
            Recorder.print_agent_turn(con=con, delay=delay)

        return self.run()

    def process_offer(self, event):
        if event.turn % 2 == 0:
            return self.process_rl_offer(event)
        else:
            return super().process_offer(event)

    def make_thread(self, priority):
        return RlThread(priority=priority,
                        rl_buyer=False,
                        thread_id=self.thread_counter)

    def is_agent_turn(self, event):
        """
        Checks whether the agent should take a turn
        :param rlenv.events.Thread.Thread event:
        :return: bool
        """
        return event.type == DELAY_EVENT and event.turn % 2 == 0

    def get_reward(self):
        raise NotImplementedError()

    def _get_max_return(self):
        return (1-self.cut) * self.lookup[START_PRICE] - LISTING_FEE

    @property
    def horizon(self):
        return 100

    @property
    def _obs_class(self):
        return SellerObs


class RelistSellerEnv(SellerEnv):

    def reset(self, next_lstg=True):
        self.init_reset(next_lstg=next_lstg)  # in AgentEnvironment
        while True:
            event, lstg_complete = super().run()  # calls EBayEnvironment.run()
            # if the lstg isn't complete that means it's time to sample an agent action
            if not lstg_complete:
                self.last_event = event
                return self.get_obs(event=event, done=False)
            # if the lstg is complete
            else:
                # check whether it's expired -- if so, relist
                if not self.outcome.sale:
                    self._relist()
                # otherwise, there's been a buy it now sale w/o a seller action,
                else:
                    # this case should happens in TestGenerator
                    # b/c lstgs with no seller actions should be removed
                    if next_lstg:
                        # conditional prevents queuing up next lstg
                        # in EvalGenerator
                        self.next_lstg()  # queue up next lstg in training
                        super().reset()
                    else:
                        return None

    def run(self):  # until EbayEnvironment.run() stops at agent turn
        while True:
            event, lstg_complete = super().run()
            if not lstg_complete or self.outcome.sale:
                return self.agent_tuple(done=lstg_complete, event=event)
            else:
                self._relist()

    def get_reward(self):
        if self.outcome is None:
            return 0.0
        else:
            gross = self.outcome.price * (1 - self.cut)
            listing_fees = LISTING_FEE * (self.relist_count + 1)
            net = gross - listing_fees
            return net

    def _relist(self):
        self.relist_count += 1
        super().reset()  # calls EBayEnvironment.reset()


class NoRelistSellerEnv(SellerEnv):

    def reset(self, next_lstg=True):
        self.init_reset(next_lstg=next_lstg)  # in AgentEnvironment
        while True:
            event, lstg_complete = super().run()  # calls EBayEnvironment.run()
            # if the lstg isn't complete that means it's time to sample an agent action
            if not lstg_complete:
                self.last_event = event
                return self.get_obs(event=event, done=False)
            # if the lstg is complete
            else:
                # this case should happens in TestGenerator
                # b/c lstgs with no seller actions should be removed
                if next_lstg:
                    # conditional prevents queuing up next lstg
                    # in EvalGenerator
                    self.next_lstg()  # queue up next lstg in training
                    super().reset()
                else:
                    return None

    def run(self):  # until EbayEnvironment.run() stops at agent turn
        while True:
            event, lstg_complete = super().run()
            return self.agent_tuple(done=lstg_complete, event=event)

    def get_reward(self):
        if self.outcome is None:
            return 0.0
        elif not self.outcome.sale:
            return - LISTING_FEE
        else:
            gross = self.outcome.price * (1 - self.cut)
            net = gross - LISTING_FEE
            return net

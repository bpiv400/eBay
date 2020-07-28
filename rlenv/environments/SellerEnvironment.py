from rlpyt.utils.collections import namedarraytuple
from featnames import START_PRICE
from constants import MAX_DELAY_TURN, POLICY_SLR, MONTH
from utils import load_sizes
from rlenv.const import DELAY_EVENT
from rlenv.environments.AgentEnvironment import AgentEnvironment
from rlenv.events.Thread import RlThread
from rlenv.generate.Recorder import Recorder

SellerObs = namedarraytuple("SellerObs",
                            list(load_sizes(POLICY_SLR)['x'].keys()))
SellerInfoTraj = namedarraytuple("SellerInfoTraj",
                                 ["months",
                                  "months_last",
                                  "bin_proceeds",
                                  "turn",
                                  "actions"])


class SellerEnvironment(AgentEnvironment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.months_last = None

    def is_agent_turn(self, event):
        """
        Checks whether the agent should take a turn
        :param rlenv.events.Thread.Thread event:
        :return: bool
        """
        return event.type == DELAY_EVENT and event.turn % 2 == 0

    def run(self):  # until EbayEnvironment.run() stops at agent turn
        while True:
            event, lstg_complete = super().run()
            if not lstg_complete:  # seller turn
                self.agent_actions += 1
            if not lstg_complete or self.outcome.sale:
                _agent_tuple = self.agent_tuple(done=lstg_complete,
                                                event=event)
                self.months_last = self._get_months(priority=event.priority)
                return _agent_tuple
            else:
                self.relist()

    def reset(self, next_lstg=True):
        self.init_reset(next_lstg=next_lstg)  # in AgentEnvironment
        while True:
            event, lstg_complete = super().run()  # calls EBayEnvironment.run()
            # if the lstg isn't complete that means it's time to sample an agent action
            if not lstg_complete:
                self.last_event = event
                self.agent_actions += 1
                self.months_last = self._get_months(priority=event.priority)
                return self.get_obs(sources=event.sources(),
                                    turn=event.turn)
            # if the lstg is complete
            else:
                # check whether it's expired -- if so, relist
                if not self.outcome.sale:
                    self.relist()
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

    def relist(self):
        self.relist_count += 1
        super().reset()  # calls EBayEnvironment.reset()

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
            delay = offer_time - self.last_event.priority
            self.last_event.prep_rl_offer(con=con, priority=offer_time)
        # expiration rejection
        else:
            delay = MAX_DELAY_TURN
            self.last_event.update_delay(seconds=MAX_DELAY_TURN)
        self.queue.push(self.last_event)
        # self.last_event = None
        if self.verbose:
            Recorder.print_agent_turn(con=con,
                                      delay=delay / MAX_DELAY_TURN)
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

    def define_observation_class(self):
        return SellerObs

    def get_reward(self):
        if self.outcome is None:
            return 0.0
        else:
            return self.outcome.price * (1-self.cut)

    def get_info(self, event=None):
        bin_proceeds = (1 - self.cut) * self.lookup[START_PRICE]
        info = SellerInfoTraj(months=self._get_months(event.priority),
                              months_last=self.months_last,
                              turn=self.last_event.turn,
                              bin_proceeds=bin_proceeds,
                              actions=self.agent_actions)
        return info

    def _get_months(self, priority=None):
        return self.relist_count + (priority - self.start_time) / MONTH

    @property
    def horizon(self):
        return 100

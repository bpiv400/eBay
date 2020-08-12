import numpy as np
from rlpyt.utils.collections import namedarraytuple
from rlenv.const import DELAY_EVENT
from rlenv.environments.AgentEnv import AgentEnv
from rlenv.events.Thread import RlThread
from rlenv.generate.Recorder import Recorder
from utils import load_sizes
from constants import MAX_DELAY_TURN, POLICY_SLR, LISTING_FEE, NUM_ACTIONS_SLR
from featnames import BYR_HIST, START_PRICE

SellerInfo = namedarraytuple("SellerInfo",
                             ["months",
                              "max_return",
                              "num_offers"])
SellerObs = namedarraytuple("SellerObs",
                            list(load_sizes(POLICY_SLR)['x'].keys()))


class SellerEnv(AgentEnv):

    def is_agent_turn(self, event):
        """
        Checks whether the agent should take a turn
        :param rlenv.events.Thread.Thread event:
        :return: bool
        """
        return event.type == DELAY_EVENT and event.turn % 2 == 0

    def reset(self, next_lstg=True):
        self.init_reset(next_lstg=next_lstg)  # in SellerEnvironment
        while True:
            event, lstg_complete = super().run()  # calls EBayEnvironment.run()

            # time to sample an agent action
            if not lstg_complete:
                self.last_event = event
                return self.get_obs(event=event, done=False)

            # if the lstg is complete
            if next_lstg:
                # conditional prevents queuing up next lstg in EvalGenerator
                self.next_lstg()  # queue up next lstg in training
                super().reset()
            else:
                return None  # for TestGenerator

    def step(self, action):
        """
        Process int giving concession/delay
        :param action: int returned from agent
        :return: tuple described in rlenv
        """
        con = self.turn_from_action(action)
        if con <= 1:
            offer_time = self.get_offer_time(self.last_event)
            delay = (offer_time - self.last_event.priority) / MAX_DELAY_TURN
            self.last_event.prep_rl_offer(con=con, priority=offer_time)
        else:
            delay = MAX_DELAY_TURN
            self.last_event.update_delay(seconds=MAX_DELAY_TURN)

        # put event in queue
        self.queue.push(self.last_event)

        # record time of offer
        self.last_event = None

        if self.verbose:
            Recorder.print_agent_turn(con=con, delay=delay)

        return self.run()

    def run(self):  # until EbayEnvironment.run() stops at agent turn
        event, lstg_complete = super().run()
        return self.agent_tuple(done=lstg_complete, event=event)

    def process_offer(self, event):
        if event.turn % 2 == 0:
            return self.process_rl_offer(event)
        else:
            return super().process_offer(event)

    def make_thread(self, priority):
        return RlThread(priority=priority,
                        rl_buyer=False,
                        thread_id=self.thread_counter)

    def get_obs(self, event=None, done=None):
        if not done:
            self.num_offers += 1
        if event.sources() is None or event.turn is None:
            raise RuntimeError("Missing arguments to get observation")
        if BYR_HIST in event.sources():
            obs_dict = self.composer.build_input_dict(model_name=None,
                                                      sources=event.sources(),
                                                      turn=event.turn)
        else:  # incomplete sources; triggers warning in AgentModel
            assert done
            obs_dict = self.empty_obs_dict
        return self._obs_class(**obs_dict)

    def get_info(self, event=None):
        max_return = (1-self.cut) * self.lookup[START_PRICE] - LISTING_FEE
        return SellerInfo(months=self._get_months(event.priority),
                          max_return=max_return,
                          num_offers=self.num_offers)

    def get_reward(self):
        if self.outcome is None:
            return 0.0
        elif not self.outcome.sale:
            return - LISTING_FEE
        else:
            gross = self.outcome.price * (1 - self.cut)
            net = gross - LISTING_FEE
            return net

    @property
    def horizon(self):
        return 100

    @property
    def _obs_class(self):
        return SellerObs

    @property
    def con_set(self):
        return np.array(range(NUM_ACTIONS_SLR)) / 100

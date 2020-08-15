from rlpyt.utils.collections import namedarraytuple
from rlenv.const import DELAY_EVENT, OFFER_EVENT
from rlenv.environments.AgentEnv import AgentEnv
from rlenv.events.Thread import RlThread
from rlenv.util import get_con_outcomes
from utils import load_sizes
from constants import POLICY_SLR
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
        if event.turn % 2 == 0:
            if event.type == DELAY_EVENT:
                return True
            if event.type == OFFER_EVENT:
                return not(self.is_lstg_expired(event) or event.thread_expired())
        return False

    def reset(self, next_lstg=True):
        self.init_reset(next_lstg=next_lstg)  # in SellerEnvironment
        while True:
            event, lstg_complete = super().run()  # calls EBayEnvironment.run()

            # time to sample an agent action
            if not lstg_complete:
                if event.type == DELAY_EVENT:  # draw delay
                    self._process_slr_delay(event)
                else:
                    self.last_event = event
                    self.prepare_offer(event)
                    return self.get_obs(event=event, done=False)

            # if the lstg is complete
            elif next_lstg:
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

        if self.verbose:
            print('AGENT TURN: con: {}'.format(con))

        con_outcomes = get_con_outcomes(con=con,
                                        sources=self.last_event.sources(),
                                        turn=self.last_event.turn)
        offer = self.last_event.update_con_outcomes(con_outcomes=con_outcomes)
        lstg_complete = self.process_post_offer(self.last_event, offer)
        if lstg_complete:
            return self.agent_tuple(event=self.last_event, done=lstg_complete)
        self.last_event = None
        return self.run()

    def run(self):  # until EbayEnvironment.run() stops at agent turn
        while True:
            event, lstg_complete = super().run()
            if event.type == DELAY_EVENT:
                self._process_slr_delay(event)
            else:
                self.last_event = event
                if not lstg_complete:
                    self.prepare_offer(event)
                return self.agent_tuple(done=lstg_complete, event=event)

    def _process_slr_delay(self, event):
        delay_seconds = self.draw_agent_delay(event)
        event.update_delay(seconds=delay_seconds)
        self.queue.push(event)
        if self.verbose:
            print('AGENT TURN: delay (sec) : {}'.format(delay_seconds))

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
        return SellerInfo(months=self._get_months(event.priority),
                          max_return=self.lookup[START_PRICE],
                          num_offers=self.num_offers)

    def get_reward(self):
        if self.outcome is None or not self.outcome.sale:
            return 0.
        return self.outcome.price

    @property
    def horizon(self):
        return 100

    @property
    def _obs_class(self):
        return SellerObs

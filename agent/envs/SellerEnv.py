from rlpyt.utils.collections import namedarraytuple
from rlenv.const import DELAY_EVENT, OFFER_EVENT
from agent.envs.AgentEnv import AgentEnv
from agent.util import define_con_set
from rlenv.util import get_delay_outcomes, get_con_outcomes
from utils import load_sizes
from constants import POLICY_SLR, MAX_DELAY_TURN
from featnames import START_PRICE

SellerInfo = namedarraytuple("SellerInfo",
                             ["days",
                              "max_return",
                              "num_offers"])
SellerObs = namedarraytuple("SellerObs",
                            list(load_sizes(POLICY_SLR)['x'].keys()))


class SellerEnv(AgentEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_delay_seconds = None

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

    def init_reset(self, next_lstg=None):
        super().init_reset(next_lstg=next_lstg)
        self.last_delay_seconds = None

    def reset(self, next_lstg=True):
        self.init_reset(next_lstg=next_lstg)  # in AgentEnvironment
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
            elif next_lstg:  # queue up next lstg in training
                self.init_reset(next_lstg=True)
            else:
                return None  # for EvalGenerator

    def step(self, action):
        """
        Process int giving concession/delay
        :param action: int returned from agent
        :return: tuple described in rlenv
        """
        con = self.turn_from_action(action)
        if self.verbose:
            print('AGENT TURN: con: {}'.format(con))

        # copy event
        thread = self.last_event
        self.last_event = None

        # execute offer
        self.num_offers += 1
        if con <= 1:
            con_outcomes = get_con_outcomes(con=con,
                                            sources=thread.sources(),
                                            turn=thread.turn)
            offer = thread.update_con_outcomes(con_outcomes)
            lstg_complete = self.process_post_offer(thread, offer)
            if lstg_complete:
                return self.agent_tuple(event=thread, done=lstg_complete)
        else:
            delay_outcomes = get_delay_outcomes(seconds=MAX_DELAY_TURN,
                                                turn=thread.turn)
            thread.sources.update_delay(delay_outcomes=delay_outcomes,
                                        turn=thread.turn)
            thread.priority += MAX_DELAY_TURN - self.last_delay_seconds
            self.queue.push(thread)
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
        self.last_delay_seconds = delay_seconds
        if self.verbose:
            print('AGENT TURN: delay (sec) : {}'.format(delay_seconds))

    def get_info(self, event=None):
        return SellerInfo(days=self._get_days(event.priority),
                          max_return=self.lookup[START_PRICE],
                          num_offers=self.num_offers)

    def get_reward(self):
        if self.outcome is None or not self.outcome.sale:
            return 0.
        return self.outcome.price

    def _define_con_set(self, con_set):
        return define_con_set(con_set=con_set, byr=False)

    @property
    def horizon(self):
        return 100

    @property
    def _obs_class(self):
        return SellerObs

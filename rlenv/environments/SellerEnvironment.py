from collections import namedtuple
from rlenv.environments import AgentEnvironment
from rlenv.env_consts import OFFER_EVENT,  ACTION_SPACE_NAME
from rlenv.env_utils import get_con_outcomes
from rlenv.spaces.ConSpace import ConSpace
from featnames import CON


class SellerEnvironment(AgentEnvironment):
    def __init__(self, params):
        super().__init__(params)

    def _is_agent_turn(self, event):
        """
        Checks whether the agent should take a turn
        :param rlenv.events.Thread.Thread event:
        :return: bool
        """
        if event.type == OFFER_EVENT and event.turn % 2 == 0:
            if self._is_lstg_expired(event):
                return False
            elif event.thread_expired():
                return False
            else:
                return True
        else:
            return False

    def step(self, action):
        """
        Process float giving concession
        :param action: float returned from agent
        :return:
        """
        offer_outcomes = get_con_outcomes(con=action, sources=self._last_event.sources(),
                                          turn=self._last_event.turn)
        offer = self._last_event.update_offer(offer_outcomes=offer_outcomes)
        lstg_complete = self._process_post_offer(self._last_event, offer)
        if lstg_complete:
            return self._agent_tuple(lstg_complete)
        self._last_event = None
        return self.run()

    def _define_action_space(self):
        # message not included because agent can't write a msg
        con = ConSpace()
        if self.agent_delay:
            nt = namedtuple(ACTION_SPACE_NAME, [CON, DELAY])
            delay = FloatBox([0.0], [1.0], null_value=0)
            return Composite([con, delay], nt)
        else:
            nt = namedtuple(ACTION_SPACE_NAME, [CON])
            return Composite([con], nt)

    def _get_reward(self):
        raise NotImplementedError("After Etan discussion")

    def _get_info(self):
        raise NotImplementedError("")






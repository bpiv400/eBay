from featnames import META
from rlenv.environments.AgentEnvironment import AgentEnvironment
from rlenv.env_consts import OFFER_EVENT, LISTING_FEE
from rlenv.env_utils import get_con_outcomes, get_cut
from agent.spaces.ConSpace import ConSpace


class SellerEnvironment(AgentEnvironment):
    def __init__(self, params):
        super(AgentEnvironment, self).__init__(params)

    def is_agent_turn(self, event):
        """
        Checks whether the agent should take a turn
        :param rlenv.events.Thread.Thread event:
        :return: bool
        """
        if event.type == OFFER_EVENT and event.turn % 2 == 0:
            if self.is_lstg_expired(event):
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
        con_outcomes = get_con_outcomes(con=action,
                                        sources=self.last_event.sources(),
                                        turn=self.last_event.turn)
        offer = self.last_event.update_con_outcomes(con_outcomes=con_outcomes)
        lstg_complete = super().process_post_offer(self.last_event, offer)
        if lstg_complete:
            return super().agent_tuple(lstg_complete)
        self.last_event = None
        return self.run()

    def define_action_space(self):
        # message not included because agent can't write a msg
        return ConSpace()

    def get_reward(self):
        if not self.last_event.is_sale():
            return 0.0
        else:
            slr_gross = self.outcome[1] * (1 - get_cut(self.lookup[META]))
            return slr_gross - LISTING_FEE

    def _get_info(self):
        raise NotImplementedError("")

    @property
    def horizon(self):
        return 100





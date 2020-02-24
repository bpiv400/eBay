from collections import namedtuple
from featnames import META, LSTG
from rlenv.environments.AgentEnvironment import AgentEnvironment
from rlenv.env_consts import OFFER_EVENT, LISTING_FEE
from rlenv.env_utils import get_con_outcomes, get_cut
from agent.spaces.ConSpace import ConSpace


class SellerEnvironment(AgentEnvironment):
    TrajInfo = namedtuple("SellerTraj", ["lstg", "relist_count", "thread",
                                        "turn", "byr_time",
                                         "byr_con", "byr_delay", "byr_msg",
                                         "slr_time", "slr_con", "slr_delay"])
    EmptyInfo = namedtuple("EmptyTraj", ["traj_done"])

    def __init__(self, **kwargs):
        super(SellerEnvironment, self).__init__(**kwargs)

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

    def run(self):
        while True:
            event, lstg_complete = super().run()
            self.last_event = event
            if not lstg_complete or self.outcome.sale:
                return self.agent_tuple(lstg_complete=lstg_complete,
                                        agent_sale=False)
            else:
                self.relist()

    # restructure
    def reset(self):
        self.reset_lstg()
        super().reset()
        while True:
            event, lstg_complete = super().run()
            # if the lstg isn't complete that means it's time to sample an agent action
            if not lstg_complete:
                self.last_event = event
                return self.composer.get_obs(sources=event.sources(),
                                             turn=event.turn)
            # if the lstg is complete
            else:
                # check whether it's expired -- if so, relist
                if not self.outcome.sale:
                    self.relist()
                # otherwise
                else:
                    self.reset_lstg()
                    super().reset()

    def relist(self):
        if self.relist_count == 0:
            self.x_lstg = self.composer.relist(x_lstg=self.x_lstg, first_lstg=False)
        self.relist_count += 1
        super().reset()

    def step(self, action):
        """
        Process float giving concession
        :param action: float returned from agent
        :return:
        """
        con = self.con_from_action(action)
        con_outcomes = get_con_outcomes(con=con,
                                        sources=self.last_event.sources(),
                                        turn=self.last_event.turn)
        offer = self.last_event.update_con_outcomes(con_outcomes=con_outcomes)
        lstg_complete = super().process_post_offer(self.last_event, offer)
        if lstg_complete:
            return super().agent_tuple(lstg_complete=lstg_complete,
                                       agent_sale=True)
        self.last_event = None
        return self.run()

    def con_from_action(self, action=None):
        return self.con_set[action]

    def define_action_space(self, con_set=None):
        # message not included because agent can't write a msg
        return ConSpace(con_set=con_set)

    def get_reward(self):
        if not self.last_event.is_sale():
            return 0.0
        else:
            slr_gross = self.outcome[1] * (1 - get_cut(self.lookup[META]))
            return slr_gross - LISTING_FEE

    def get_info(self, agent_sale=False, lstg_complete=False):
        # initialize vars
        tuple_dict = {
            "lstg": self.lookup[LSTG],
            "thread": self.last_event.thread_id,
            "relist_count": 0,  # TODO UPDATE
        }

        if not agent_sale:
            tuple_dict['turn'] = self.last_event.turn - 1
            if self.last_event.turn == 2:
                tuple_dict['slr_time'] = None
                tuple_dict['slr_delay'] = None
                tuple_dict['slr_con'] = None
        return self.EmptyInfo(traj_done=lstg_complete)

    @property
    def horizon(self):
        return 100





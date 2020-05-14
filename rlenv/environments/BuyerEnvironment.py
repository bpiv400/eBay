"""
Environment for training the buyer agent
"""
from rlenv.env_consts import DELAY_EVENT, RL_ARRIVE_EVENT
from rlenv.environments.AgentEnvironment import AgentEnvironment


class BuyerEnvironment(AgentEnvironment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def turn_from_action(self, action=None):  # might not make sense
        pass

    def get_reward(self):
        pass

    @property
    def horizon(self):
        pass

    def step(self, action):
        pass

    def is_agent_turn(self, event):
        if event.type == RL_ARRIVE_EVENT:
            return not (self.is_lstg_expired(event))
        else:
            return event.type == DELAY_EVENT and event.turn % 2 == 1

    def define_action_space(self, con_set=None):
        return
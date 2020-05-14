"""
Environment for training the buyer agent
"""
from rlenv.env_consts import DELAY_EVENT, RL_ARRIVAL_EVENT
from rlenv.env_utils import get_clock_feats
from rlenv.sources import RlArrivalSources
from rlenv.environments.AgentEnvironment import AgentEnvironment
from rlenv.events.Arrival import Arrival


class BuyerEnvironment(AgentEnvironment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def turn_from_action(self, action=None):  # might not make sense
        return self.con_set[action]

    def get_reward(self):
        pass

    @property
    def horizon(self):
        return 100

    def step(self, action):
        con = self.turn_from_action(action=action)
        if self.last_event.event_type == RL_ARRIVAL_EVENT:
            if con == 0:
                # delay
            else:
                # concession and new thread
        else:
            if con == 0:
                # expiration rejection and end the simulation
            else:
                if con != 1:
                    # ordinary concession
                else:
                    # accept the seller's last offer and end the  simulation



    def reset(self):
        """
        Resets the environment by drawing a new listing,
        resetting the queue, adding a buyer rl arrival event, then
        running the environment
        :return: observation associated with the first rl arrival
        """
        self.reset_lstg()
        super().reset()
        rl_sources = RlArrivalSources(x_lstg=self.x_lstg)
        event = Arrival(priority=self.start_time, sources=rl_sources)
        self.queue.push(event)
        # should deterministically return RL_ARRIVAL_EVENT at start of lstg
        # lstg should never be complete at this point
        event, _ = super().run()
        event.update_arrival()
        self.last_event = event
        return self.get_obs()

    def is_agent_turn(self, event):
        """
        Indicates the buyer agent needs to take a turn when there's an
        agent arrival event or when it's buyer's turn in an RL thread
        :return:
        """
        if event.type == RL_ARRIVAL_EVENT:
            return not (self.is_lstg_expired(event))
        else:
            return (event.type == DELAY_EVENT and
                    event.turn % 2 == 1 and event.rl)

    def define_action_space(self, con_set=None):
        return
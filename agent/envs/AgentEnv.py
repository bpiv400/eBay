import numpy as np
import torch
from rlpyt.envs.base import Env
from rlpyt.spaces.composite import Composite
from rlpyt.spaces.float_box import FloatBox
from rlenv.EBayEnv import EBayEnv
from rlenv.events.Thread import RlThread
from agent.ConSpace import ConSpace
from constants import INTERVAL_TURN, INTERVAL_CT_TURN, MONTH, CON_MULTIPLIER


class AgentEnv(EBayEnv, Env):
    def __init__(self, **kwargs):
        super().__init__(params=kwargs)
        self.last_event = None
        self.num_offers = None  # number of agent offers (excl. byr delays)
        self.empty_obs_dict = {k: torch.zeros(v).float()
                               for k, v in self.composer.agent_sizes['x'].items()}

        # action space
        self.con_set = np.array(range(CON_MULTIPLIER + 1)) / 100
        self._action_space = self._define_action_space()

        # observation space
        self._observation_space = self.define_observation_space()

    def define_observation_space(self):
        sizes = self.composer.agent_sizes['x']
        boxes = [FloatBox(-1000, 1000, shape=size) for size in sizes.values()]
        return Composite(boxes, self._obs_class)

    def agent_tuple(self, event=None, done=None):
        """
        Constructs observation and calls child environment to get reward
        and info, then sets self.last_event to current event.
        :param RlThread event: either agent's turn or trajectory is complete.
        :param bool done: True if trajectory complete.
        :return: tuple
        """
        obs = self.get_obs(event=event, done=done)
        reward = self.get_reward()
        info = self.get_info(event=event)
        return obs, reward, done, info

    def get_obs(self, event=None, done=None):
        raise NotImplementedError()

    def get_reward(self):
        raise NotImplementedError()

    def get_info(self, event=None):
        raise NotImplementedError()

    def draw_agent_delay(self, event):
        # query delay model
        input_dict = self.get_delay_input_dict(event=event)
        intervals = (self.end_time - event.priority) / INTERVAL_TURN
        max_interval = min(int(intervals), INTERVAL_CT_TURN)
        delay_seconds = self.get_delay(input_dict=input_dict,
                                       turn=event.turn,
                                       thread_id=event.thread_id,
                                       time=event.priority,
                                       max_interval=max(1, max_interval))
        return max(1, delay_seconds)

    def init_reset(self, next_lstg=True):
        self.last_event = None
        self.num_offers = 0
        if next_lstg:
            if not self.has_next_lstg():
                raise RuntimeError("Out of lstgs")
            self.next_lstg()
        super().reset()  # calls EBayEnvironment.reset()

    def turn_from_action(self, action=None):
        return self.con_set[action]

    @property
    def horizon(self):
        return NotImplementedError()

    @property
    def _obs_class(self):
        raise NotImplementedError()

    def _define_action_space(self):
        return ConSpace(size=len(self.con_set))

    def is_agent_turn(self, event):
        raise NotImplementedError()

    def step(self, action):
        """
        Process float giving concession
        :param action: float returned from agent
        :return:
        """
        raise NotImplementedError()

    def _get_months(self, priority=None):
        return self.relist_count + (priority - self.start_time) / MONTH

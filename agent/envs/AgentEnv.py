import torch
from rlpyt.envs.base import Env
from rlpyt.spaces.composite import Composite
from rlpyt.spaces.float_box import FloatBox
from rlpyt.utils.collections import namedarraytuple
from rlenv.EBayEnv import EBayEnv
from rlenv.events.Thread import Thread
from agent.ConSpace import ConSpace
from constants import INTERVAL_TURN, INTERVAL_CT_TURN, DAY, MAX_DELAY_TURN
from featnames import BYR_HIST

Info = namedarraytuple("Info", ["days", "max_return",
                                "num_delays", "num_offers",
                                "turn", "thread_id", "priority"])


class AgentEnv(EBayEnv, Env):
    def __init__(self, **kwargs):
        super().__init__(params=kwargs)
        self.test = False if 'test' not in kwargs else kwargs['test']

        # parameters to be set later
        self.last_event = None
        self.item_value = None
        self.num_delays = None  # only relevant for byr agents
        self.num_offers = None  # number of agents offers (excl. byr delays)

        # for passing an empty observation to agents
        self.empty_dict = {k: torch.zeros(v).float()
                           for k, v in self.composer.agent_sizes['x'].items()}

        # action space
        self.con_set = self._define_con_set(kwargs['con_set'])
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
        :param Thread event: either agents's turn or trajectory is complete.
        :param bool done: True if trajectory complete.
        :return: tuple
        """
        obs = self.get_obs(event=event, done=done)
        reward = self.get_reward()
        info = self.get_info(event=event)
        return obs, reward, done, info

    def get_obs(self, event=None, done=None):
        if event.sources() is None or event.turn is None:
            raise RuntimeError("Missing arguments to get observation")
        if BYR_HIST in event.sources():
            obs_dict = self.composer.build_input_dict(model_name=None,
                                                      sources=event.sources(),
                                                      turn=event.turn)
        else:  # incomplete sources; triggers warning in AgentModel
            assert done
            obs_dict = self.empty_dict
        return self._obs_class(**obs_dict)

    def get_reward(self):
        raise NotImplementedError()

    def get_info(self, event=None):
        thread_id = 0 if not isinstance(event, Thread) else event.thread_id
        return Info(days=self._get_days(event.priority),
                    max_return=self.item_value,
                    num_delays=self.num_delays,
                    num_offers=self.num_offers,
                    turn=event.turn,
                    thread_id=thread_id,
                    priority=event.priority)

    def draw_agent_delay(self, event):
        # query delay model
        input_dict = self.get_delay_input_dict(event=event)
        intervals = (self.end_time - event.priority) / INTERVAL_TURN
        max_interval = max(1, min(int(intervals), INTERVAL_CT_TURN))
        delay_seconds = self.get_delay(input_dict=input_dict,
                                       turn=event.turn,
                                       thread_id=event.thread_id,
                                       time=event.priority,
                                       max_interval=max_interval)
        if not self.test:  # expiration delays only allowed in testing
            assert delay_seconds < MAX_DELAY_TURN
        return max(1, delay_seconds)

    def init_reset(self, next_lstg=True):
        self.last_event = None
        self.num_delays = 0
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

    def _define_con_set(self, con_set):
        raise NotImplementedError()

    def is_agent_turn(self, event):
        raise NotImplementedError()

    def step(self, action):
        """
        Process float giving concession
        :param action: float returned from agents
        :return:
        """
        raise NotImplementedError()

    def _get_days(self, priority=None):
        return (priority - self.start_time) / DAY

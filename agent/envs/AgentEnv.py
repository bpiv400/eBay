from collections import namedtuple
import numpy as np
import torch
from rlpyt.envs.base import Env
from rlpyt.spaces.composite import Composite
from rlpyt.spaces.float_box import FloatBox
from rlpyt.utils.collections import namedarraytuple
from rlenv.EBayEnv import EBayEnv
from rlenv.events.Thread import Thread
from agent.ConSpace import ConSpace
from agent.const import AGENT_CONS
from constants import INTERVAL, INTERVAL_CT_TURN, DAY, NUM_COMMON_CONS, IDX
from featnames import START_PRICE, BYR, SLR, DELTA, TURN_COST

Info = namedarraytuple("Info", ["days", "max_return", "num_actions", "turn",
                                "thread_id", "priority", "agent_sale"])
EventLog = namedtuple("EventLog", ["priority", "thread_id", "turn"])


class AgentEnv(EBayEnv, Env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.byr = kwargs[BYR]

        if not self.test:  # for reward calculation
            self.delta = kwargs[DELTA]

        if self.train:
            self.turn_cost = kwargs[TURN_COST]

        # parameters to be set later
        self.curr_event = None
        self.last_event = None
        self.num_actions = None  # number of agent actions

        # for passing an empty observation to agents
        self.empty_dict = {k: torch.zeros(v).float()
                           for k, v in self.composer.agent_sizes['x'].items()}

        # action space
        self.con_set = self._define_con_set()
        self._action_space = self._define_action_space()

        # observation space
        self._observation_space = self.define_observation_space()

    def define_observation_space(self):
        sizes = self.composer.agent_sizes['x']
        boxes = [FloatBox(-1000, 1000, shape=size) for size in sizes.values()]
        return Composite(boxes, self._obs_class)

    def agent_tuple(self, event=None, done=None, last_event=None):
        """
        Constructs observation and calls child environment to get reward
        and info, then sets self.last_event to current event.
        :param Thread event: either agent's turn or trajectory is complete.
        :param bool done: True if trajectory complete.
        :param EventLog last_event: for constructing info tuple.
        :return: tuple
        """
        obs = self.get_obs(event=event, done=done)
        if not self.train:
            reward, agent_sale = None, None
        else:
            reward, agent_sale = self.get_reward()
            if self.verbose and done:
                print('Agent reward: ${0:.2f}. Normalized: {1:.1f}%'.format(
                    reward, 100 * reward / self.lookup[START_PRICE]))
        info = self.get_info(event=last_event, agent_sale=agent_sale)
        return obs, reward, done, info

    def get_obs(self, event=None, done=None):
        if not done:
            if event.sources() is None or event.turn is None:
                raise RuntimeError("Missing arguments to get observation")
            obs_dict = self.composer.build_input_dict(model_name=None,
                                                      sources=event.sources(),
                                                      turn=event.turn)
        else:
            obs_dict = self.empty_dict
        return self._obs_class(**obs_dict)

    def get_reward(self):
        raise NotImplementedError()

    def get_info(self, event=None, agent_sale=None):
        return Info(days=self._get_days(event.priority),
                    max_return=self.lookup[START_PRICE],
                    num_actions=self.num_actions,
                    turn=event.turn,
                    thread_id=event.thread_id,
                    priority=event.priority,
                    agent_sale=agent_sale)

    def draw_agent_delay(self, event):
        input_dict = self.get_delay_input_dict(event=event)
        intervals = (self.end_time - event.priority) / INTERVAL
        max_interval = max(1, min(int(intervals), INTERVAL_CT_TURN))
        delay_seconds = self.query_strategy.get_delay(
            input_dict=input_dict,
            turn=event.turn,
            thread_id=event.thread_id,
            time=event.priority,
            max_interval=max_interval
        )
        return max(1, delay_seconds)

    def init_reset(self):
        self.curr_event = None
        self.last_event = None
        self.num_actions = 0
        if self.train:
            if not self.has_next_lstg():
                raise RuntimeError("Out of lstgs")
            self.next_lstg()
        super().reset()  # calls EBayEnv.reset()

    def turn_from_action(self, turn=None, action=None):
        return self.con_set[turn][action]

    @property
    def horizon(self):
        return NotImplementedError()

    @property
    def _obs_class(self):
        raise NotImplementedError()

    def _define_con_set(self):
        if self.test:
            cons = np.arange(101) / 100
            cons = {t: cons for t in range(1, 7)}
        elif self.byr:
            cons = {t: AGENT_CONS[t] for t in IDX[BYR][:-1]}
        else:
            cons = {t: AGENT_CONS[t] for t in IDX[SLR]}
        return cons

    def _define_action_space(self):
        if self.test:
            num_cons = 101
        else:
            num_cons = NUM_COMMON_CONS + (2 if self.byr else 3)
        return ConSpace(size=num_cons)

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

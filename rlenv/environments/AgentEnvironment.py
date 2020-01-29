import pandas as pd
import numpy as np
from rlpyt.envs.base import Env
from rlpyt.spaces.composite import Composite
from rlpyt.spaces.float_box import FloatBox
from featnames import START_TIME
from constants import MONTH
from rlenv.env_consts import (LOOKUP, X_LSTG, ENV_LSTG_COUNT)
from rlenv.Composer import AgentComposer
from rlenv.environments.EbayEnvironment import EbayEnvironment


class AgentEnvironment(EbayEnvironment, Env):

    def __init__(self, params):
        super().__init__(params)
        # attributes for getting lstg data
        self._file = params['file']
        self._num_lstgs = len(self._file[LOOKUP])
        self._lookup_cols = self._file[LOOKUP].attrs['cols']
        self._lookup_cols = [col.decode('utf-8') for col in self._lookup_cols]
        self._lookup_slice, self._x_lstg_slice = None, None
        self._ix = -1

        # model interfaces and composer
        self._composer = params['composer']  # type: AgentComposer
        self.last_event = None  # type: Thread
        # action and observation spaces
        self._action_space = self._define_action_space()
        self._observation_space = self._define_observation_space()

    def reset(self):
        while True:
            self._reset_lstg()
            super().reset()
            event, lstg_complete = super().run()
            if not lstg_complete:
                self.last_event = event
                return self._composer.get_obs(sources=event.sources(),
                                              turn=event.turn)

    def run(self):
        event, lstg_complete = super().run()
        self.last_event = event
        return self._agent_tuple(lstg_complete)

    def _define_observation_space(self):
        sizes = self._composer.agent_sizes
        boxes = [FloatBox(-1000, 1000, shape=len(size)) for size in sizes.values()]
        return Composite(boxes, self._composer.obs_space_class)

    def _reset_lstg(self):
        """
        Sample a new lstg from the file and set lookup and x_lstg series
        """
        if self._ix == -1 or self._ix == self._num_lstgs:
            self._draw_lstgs()
        self.x_lstg = pd.Series(self._x_lstg_slice[self._ix, :], index=self._composer.x_lstg_cols)
        self.x_lstg = self._composer.decompose_x_lstg(self.x_lstg)
        self.lookup = pd.Series(self._lookup_slice[self._ix, :], index=self._lookup_cols)
        self._ix += 1
        self.end_time = self.lookup[START_TIME] + MONTH

    def _draw_lstgs(self):
        ids = np.random.randint(0, self._num_lstgs, ENV_LSTG_COUNT)
        self._lookup_slice = self._file[LOOKUP][ids, :]
        self._x_lstg_slice = self._file[X_LSTG][ids, :]
        self._ix = 0

    def _agent_tuple(self, lstg_complete):
        obs = self._composer.get_obs(sources=self.last_event.sources(),
                                     turn=self.last_event.turn)
        return obs, self._get_reward(), lstg_complete, self._get_info()

    def _get_reward(self):
        raise NotImplementedError("After Etan discussion")

    def _get_info(self):
        raise NotImplementedError("")

    @property
    def horizon(self):
        raise NotImplementedError()

    def step(self, action):
        """
        Process float giving concession
        :param action: float returned from agent
        :return:
        """
        raise NotImplementedError()

    def _record(self, event, start_thread=None, byr_hist=None):
        raise NotImplementedError("Double check method signature")

    def _is_agent_turn(self, event):
        raise NotImplementedError()

    def _define_action_space(self):
        raise NotImplementedError()



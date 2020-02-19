import os
import h5py
import pandas as pd
import numpy as np
from rlpyt.envs.base import Env
from rlpyt.spaces.composite import Composite
from rlpyt.spaces.float_box import FloatBox
from featnames import START_TIME
from constants import MONTH
from rlenv.env_consts import (LOOKUP, X_LSTG, ENV_LSTG_COUNT)
from rlenv.environments.EbayEnvironment import EbayEnvironment
from agent.agent_utils import get_con_set
from agent.agent_consts import CON_TYPE


class AgentEnvironment(EbayEnvironment, Env):

    def __init__(self, **kwargs):
        super().__init__(params=kwargs)
        # attributes for getting lstg data
        self._filename = kwargs['filename']
        self._file = self.open_input_file()
        self._num_lstgs = len(self._file[LOOKUP])
        self._lookup_cols = self._file[LOOKUP].attrs['cols']
        self._lookup_cols = [col.decode('utf-8') for col in self._lookup_cols]
        self._lookup_slice, self._x_lstg_slice = None, None
        self._ix = -1
        
        self.last_event = None  # type: Thread
        # action and observation spaces
        self.con_set = get_con_set(self.composer.con_type)
        self._action_space = self.define_action_space(con_set=self.con_set)
        self._observation_space = self.define_observation_space()

    def open_input_file(self):
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        f = h5py.File(self._filename, "r")
        return f

    def reset(self):
        while True:
            self._reset_lstg()
            super().reset()
            event, lstg_complete = super().run()
            if not lstg_complete:
                self.last_event = event
                return self.composer.get_obs(sources=event.sources(),
                                             turn=event.turn)

    def run(self):
        event, lstg_complete = super().run()
        self.last_event = event
        return self.agent_tuple(lstg_complete)

    def define_observation_space(self):
        sizes = self.composer.agent_sizes['x']
        boxes = [FloatBox(-1000, 1000, shape=len(size)) for size in sizes.values()]
        return Composite(boxes, self.composer.obs_space_class)

    def _reset_lstg(self):
        """
        Sample a new lstg from the file and set lookup and x_lstg series
        """
        if self._ix == -1 or self._ix == self._num_lstgs:
            self._draw_lstgs()
        self.x_lstg = pd.Series(self._x_lstg_slice[self._ix, :], index=self.composer.x_lstg_cols)
        self.x_lstg = self.composer.decompose_x_lstg(self.x_lstg)
        self.lookup = pd.Series(self._lookup_slice[self._ix, :], index=self._lookup_cols)
        self._ix += 1
        self.start_time = self.lookup[START_TIME]
        self.end_time = self.start_time + MONTH

    def _draw_lstgs(self):
        ids = np.random.randint(0, self._num_lstgs, ENV_LSTG_COUNT)
        self._lookup_slice = self._file[LOOKUP][ids, :]
        self._x_lstg_slice = self._file[X_LSTG][ids, :]
        self._ix = 0

    def agent_tuple(self, lstg_complete):
        obs = self.composer.get_obs(sources=self.last_event.sources(),
                                    turn=self.last_event.turn)
        return obs, self.get_reward(), lstg_complete, self._get_info()

    def con_from_action(self, action=None):
        raise NotImplementedError("")

    def get_reward(self):
        raise NotImplementedError("")

    def _get_info(self):
        return None

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

    # TODO: May update
    def record(self, event, start_thread=None, byr_hist=None):
        pass

    def is_agent_turn(self, event):
        raise NotImplementedError()

    def define_action_space(self, con_set=None):
        raise NotImplementedError()



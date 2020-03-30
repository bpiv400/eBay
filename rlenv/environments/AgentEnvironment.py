import os
import h5py
import pandas as pd
import numpy as np
from collections import namedtuple
from rlpyt.envs.base import Env
from rlpyt.spaces.composite import Composite
from rlpyt.spaces.float_box import FloatBox
from featnames import START_TIME
from constants import MONTH
from rlenv.env_consts import (LOOKUP, X_LSTG, ENV_LSTG_COUNT)
from rlenv.environments.EbayEnvironment import EbayEnvironment
from rlenv.simulator.Recorder import Recorder
from agent.agent_utils import get_con_set
from agent.agent_consts import seller_groupings


SellerObs = namedtuple("SellerObs", seller_groupings)


class AgentEnvironment(EbayEnvironment, Env):

    def __init__(self, **kwargs):
        super().__init__(params=kwargs)
        # attributes for getting lstg data
        self._filename = kwargs['filename']
        self._file = None
        self._file_opened = False
        self._num_lstgs = None
        self._lookup_cols = None
        self._lookup_slice, self._x_lstg_slice = None, None
        self._ix = -1
        self.relist_count = 0
        
        self.last_event = None  # type: Thread
        # action and observation spaces
        self.con_set = get_con_set(self.composer.con_type)
        self._action_space = self.define_action_space(con_set=self.con_set)
        self._observation_space = self.define_observation_space()

    def open_input_file(self):
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        self._file = h5py.File(self._filename, "r")
        self._num_lstgs = len(self._file[LOOKUP])
        self._lookup_cols = self._file[LOOKUP].attrs['cols']
        self._lookup_cols = [col.decode('utf-8') for col in self._lookup_cols]
        self._file_opened = True

    def define_observation_space(self):
        sizes = self.composer.agent_sizes['x']
        boxes = [FloatBox(-1000, 1000, shape=size) for size in sizes.values()]
        return Composite(boxes, SellerObs)

    def reset_lstg(self):
        """
        Sample a new lstg from the file and set lookup and x_lstg series
        """
        if not self._file_opened:
            self.open_input_file()
        if self._ix == -1 or self._ix == ENV_LSTG_COUNT:
            self._draw_lstgs()
        self.x_lstg = pd.Series(self._x_lstg_slice[self._ix, :], index=self.composer.x_lstg_cols)
        self.x_lstg = self.composer.decompose_x_lstg(self.x_lstg)
        self.lookup = pd.Series(self._lookup_slice[self._ix, :], index=self._lookup_cols)
        self._ix += 1
        self.start_time = self.lookup[START_TIME]
        self.end_time = self.start_time + MONTH
        self.relist_count = 0
        self.x_lstg = self.composer.relist(x_lstg=self.x_lstg, first_lstg=True)

    def _draw_lstgs(self):
        ids = np.random.choice(self._num_lstgs, ENV_LSTG_COUNT,
                               replace=False)
        reordering = np.argsort(ids)
        sorted_ids = ids[reordering]
        unsorted_ids = np.argsort(reordering)
        self._lookup_slice = self._file[LOOKUP][sorted_ids, :]
        self._x_lstg_slice = self._file[X_LSTG][sorted_ids, :]
        self._lookup_slice = self._lookup_slice[unsorted_ids, :]
        self._x_lstg_slice = self._x_lstg_slice[unsorted_ids, :]
        self._ix = 0

    def agent_tuple(self, lstg_complete=None, agent_sale=None):
        obs = self.get_obs(sources=self.last_event.sources(),
                           turn=self.last_event.turn)
        return (obs, self.get_reward(), lstg_complete,
                self.get_info(agent_sale=agent_sale, lstg_complete=lstg_complete))

    def get_obs(self, sources=None, turn=None):
        obs_dict = self.composer.get_obs(sources=sources, turn=turn)
        return SellerObs(**obs_dict)

    def con_from_action(self, action=None):
        raise NotImplementedError()

    def get_reward(self):
        raise NotImplementedError()

    def get_info(self, agent_sale=False, lstg_complete=False):
        return NotImplementedError()

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
    def record(self, event, byr_hist=None, censored=False):
        if not censored and byr_hist is None:
            # print('summary in record')
            # print(event.summary())
            if self.verbose:
                Recorder.print_offer(event)

    def is_agent_turn(self, event):
        raise NotImplementedError()

    def define_action_space(self, con_set=None):
        raise NotImplementedError()



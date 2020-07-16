import numpy as np
from compress_pickle import load
from featnames import START_TIME, META, X_LSTG, LOOKUP, P_ARRIVAL
from constants import MONTH, AGENT_PARTS_DIR, TRAIN_RL, \
    INTERVAL_TURN, INTERVAL_CT_TURN
from utils import get_months_since_lstg, get_cut
from agent.ConSpace import ConSpace
from rlpyt.envs.base import Env
from rlpyt.spaces.composite import Composite
from rlpyt.spaces.float_box import FloatBox
from rlenv.environments.EbayEnvironment import EbayEnvironment
from rlenv.generate.Recorder import Recorder


class AgentEnvironment(EbayEnvironment, Env):
    def __init__(self, **kwargs):
        super().__init__(params=kwargs)
        # attributes for getting lstg data
        if 'rank' not in kwargs:
            filename = self._get_train_file_path(rank=0)
        else:
            filename = self._get_train_file_path(rank=kwargs['rank'])
        self._file = load(filename)
        self._num_lstgs = len(self._file[LOOKUP])
        self._ix = -1
        self._lstgs = None

        self.relist_count = 0
        self.last_event = None  # type: Thread

        # action space
        num_actions = kwargs['composer'].sizes['agent']['out']
        self.con_set = np.array(range(num_actions)) / 100
        self._action_space = self.define_action_space()

        # observation space
        self._obs_class = self.define_observation_class()
        self._observation_space = self.define_observation_space()

        self.cut = None

    def define_observation_class(self):
        raise NotImplementedError("Please extend")

    def define_observation_space(self):
        sizes = self.composer.agent_sizes['x']
        boxes = [FloatBox(-1000, 1000, shape=size) for size in sizes.values()]
        return Composite(boxes, self._obs_class)

    def reset_lstg(self):
        """
        Sample a new lstg from the file and set lookup and x_lstg series
        """
        if self._ix == -1 or self._ix == self._num_lstgs:
            self._draw_lstgs()
        lstg = self._lstgs[self._ix]
        self.x_lstg = self._file[X_LSTG].loc[lstg, :]
        self.x_lstg = self.composer.decompose_x_lstg(self.x_lstg)
        self.lookup = self._file[LOOKUP].loc[lstg, :]
        self.p_arrival = self._file[P_ARRIVAL].loc[lstg, :].values
        self._ix += 1
        self.start_time = self.lookup[START_TIME]
        self.end_time = self.start_time + MONTH
        self.relist_count = 0
        self.cut = get_cut(self.lookup[META])
        if self.verbose:
            Recorder.print_lstg(self.lookup)

    @staticmethod
    def _get_train_file_path(rank=None):
        return AGENT_PARTS_DIR + '{}/chunks/{}.gz'.format(TRAIN_RL, rank)

    def _draw_lstgs(self):
        self._lstgs = np.array(self._file[LOOKUP].index)
        np.random.shuffle(self._lstgs)
        self._ix = 0

    def agent_tuple(self, done=None):
        obs = self.get_obs(sources=self.last_event.sources(),
                           turn=self.last_event.turn)
        reward = self.get_reward()
        info = self.get_info()
        return obs, reward, done, info

    def get_obs(self, sources=None, turn=None):
        if sources is None or turn is None:
            raise RuntimeError("Missing arguments to get observation")
        obs_dict = self.composer.build_input_dict(model_name=None,
                                                  sources=sources,
                                                  turn=turn)
        return self._obs_class(**obs_dict)

    def get_reward(self):
        raise NotImplementedError()

    def get_info(self):
        raise NotImplementedError()

    def get_offer_time(self, event):
        # query with delay model
        input_dict = self.get_delay_input_dict(event=event)
        intervals = (self.end_time - event.priority) / INTERVAL_TURN
        max_interval = min(int(intervals), INTERVAL_CT_TURN)
        delay = self.get_delay(input_dict=input_dict,
                               turn=event.turn,
                               thread_id=event.thread_id,
                               time=event.priority,
                               max_interval=max(1, max_interval))
        return max(delay, 1) + event.priority

    def process_rl_offer(self, event):
        """
        :param RlThread event:
        :return: bool indicating the lstg is over
        """
        # check whether the lstg expired, censoring this offer
        if self.is_lstg_expired(event):
            return self.process_lstg_expiration(event)
        slr_offer = event.turn % 2 == 0
        if event.thread_expired():
            if slr_offer:
                self.process_slr_expire(event)
                return False
            else:
                raise RuntimeError("Thread should never expire before"
                                   "buyer agent offer")
        time_feats = self.time_feats.get_feats(thread_id=event.thread_id,
                                               time=event.priority)
        months_since_lstg = None
        if event.turn == 1:
            months_since_lstg = get_months_since_lstg(lstg_start=self.start_time,
                                                      time=event.priority)
        event.init_rl_offer(months_since_lstg=months_since_lstg, time_feats=time_feats)
        offer = event.execute_offer()
        return self.process_post_offer(event, offer)

    def turn_from_action(self, action=None):
        return self.con_set[action]

    @property
    def horizon(self):
        return NotImplementedError()

    def define_action_space(self):
        return ConSpace(size=len(self.con_set))

    # TODO: may update
    def record(self, event, byr_hist=None, censored=False):
        if not censored and byr_hist is None:
            if self.verbose:
                Recorder.print_offer(event)

    def is_agent_turn(self, event):
        raise NotImplementedError()

    def step(self, action):
        """
        Process float giving concession
        :param action: float returned from agent
        :return:
        """
        raise NotImplementedError()




"""
Runner for use with PPO (later EBayPPO)
"""

import time
from collections import deque
from rlpyt.utils.seed import set_seed, make_seed
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging import logger
from rlpyt.utils.logging.logger import _tabular


class EbayRunner(MinibatchRl):
    def __init__(self,
                 algo,
                 agent,
                 sampler,
                 seed=None,
                 affinity=None,
                 batches_per_eval=None,
                 batch_size=None):
        # initialize logging
        self._opt_infos = None
        self._traj_infos = None
        self._cum_time = 0
        self._cum_completed_trajs = 0
        self._new_completed_trajs = 0
        self._last_update_counter = 0
        self._last_time = 0
        # super constructor
        super().__init__(algo=algo, agent=agent, sampler=sampler,
                         seed=seed, affinity=affinity,
                         n_steps=batch_size * batches_per_eval,
                         log_interval_steps=batch_size)
        # re initializing components
        self.algo = algo
        self.sampler = sampler
        self.agent = agent
        # additional state vars
        self.batches_per_eval = batches_per_eval
        self.itr_ = 0

    def update_agent(self, agent):
        self.agent = agent

    def update_algo(self, algo):
        self.algo = algo

    def train(self):
        n_itr = self.startup()
        set_seed(make_seed())
        assert self.log_interval_itrs == 1
        assert n_itr == self.batches_per_eval
        for itr in range(n_itr):
            logger.set_iteration(itr + self.itr_)
            with logger.prefix(f"itr #{itr + self.itr_} "):
                self.agent.sample_mode(itr + self.itr_)  # Might not be this agent sampling.
                samples, traj_infos = self.sampler.obtain_samples(itr + self.itr_)
                self.agent.train_mode(itr + self.itr_)
                opt_info = self.algo.optimize_agent(itr + self.itr_, samples)
                self.store_diagnostics(itr + self.itr_, traj_infos, opt_info)
                self.log_diagnostics(itr + self.itr_)

        self.itr_ += self.batches_per_eval
        self.shutdown()
        return self.itr_

    def initialize_logging(self):
        self._traj_infos = deque(maxlen=self.log_traj_window)
        if self.itr_ == 0:
            self._opt_infos = {k: list() for k in self.algo.opt_info_fields}
        self._last_time = time.time()

    def shutdown(self):
        self.sampler.shutdown()

    def log_diagnostics(self, itr, traj_infos=None, eval_time=0):
        """
        Write diagnostics (including stored ones) to csv via the logger.
        """
        # save model if evaluation happens after this batch
        if (itr + 1) % self.batches_per_eval == 0:
            self.save_itr_snapshot(itr)

        # update cumulative variables
        new_time = time.time()
        self._cum_time += (new_time - self._last_time)
        new_updates = self.algo.update_counter - self._last_update_counter
        new_samples = (self.sampler.batch_size * self.world_size *
                       self.log_interval_itrs)
        updates_per_second = (new_updates / self._cum_time)
        samples_per_second = (new_samples / self._cum_time)

        # log diagnostics
        logger.record_tabular('CumTrainTime', self._cum_time)
        logger.record_tabular('Iteration', itr)
        logger.record_tabular('NewCompletedTrajs', self._new_completed_trajs)
        logger.record_tabular('CumCompletedTrajs', self._cum_completed_trajs)
        logger.record_tabular('StepsPerSecond', samples_per_second)
        logger.record_tabular('UpdatesPerSecond', updates_per_second)
        self._log_infos(traj_infos)
        # Is not dumping at any point problematic
        # logger.dump_tabular(with_prefix=False)
        del _tabular[:]
        self._last_time = new_time
        self._last_update_counter = self.algo.update_counter

    def store_diagnostics(self, itr, traj_infos, opt_info):
        """
        Store any diagnostic information from a training iteration that should
        be kept for the next logging iteration.
        """
        self._new_completed_trajs = len(traj_infos)
        self._traj_infos.extend(traj_infos)
        self._cum_completed_trajs += len(traj_infos)
        for k, v in self._opt_infos.items():
            new_v = getattr(opt_info, k, [])
            v.extend(new_v if isinstance(new_v, list) else [new_v])

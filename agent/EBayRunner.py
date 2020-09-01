import psutil
import time
import torch
import numpy as np
from collections import deque
from rlpyt.runners.base import BaseRunner
from rlpyt.samplers.parallel.base import ParallelSamplerBase
from rlpyt.samplers.parallel.worker import initialize_worker
from rlpyt.utils.collections import AttrDict
from rlpyt.utils.seed import set_seed, make_seed, set_envs_seeds
from rlpyt.utils.logging import logger
from rlpyt.utils.prog_bar import ProgBarCounter
from rlenv.LstgLoader import TrainLoader


class EBayBaseRunner(BaseRunner):
    """
    Implements startup, logging, and agents checkpointing functionality, to be
    called in the `train()` method of the subclassed runner.  Subclasses will
    modify/extend many of the methods here.
    Args:
        algo: The algorithm instance.
        agent: The learning agents instance.
        sampler: The sampler instance.
        seed (int): Random seed to use, if ``None`` will generate randomly.
        affinity (dict): Hardware component assignments for sampler and algorithm.
    """
    def __init__(
            self,
            algo,
            agent,
            sampler,
            seed=None,
            affinity=None,
    ):
        # save parameters to self
        self.algo = algo
        self.agent = agent
        self.sampler = sampler
        self.seed = seed
        self.affinity = dict() if affinity is None else affinity
        self.rank = 0
        self.log_interval_itrs = 1

        # other fixed parameters
        self.min_itr_learn = getattr(self.algo, 'min_itr_learn', 0)
        self._opt_infos = {k: list() for k in self.algo.opt_info_fields}

        # parameters to be set later
        self.itr_batch_size = None
        self._start_time = None
        self._last_time = None
        self._cum_time = None
        self._cum_completed_trajs = None
        self._last_update_counter = None
        self._traj_infos = None
        self._new_completed_trajs = None
        self.pbar = None

    def startup(self):
        """
        Sets hardware affinities, initializes the following: 1) sampler (which
        should initialize the agents), 2) agents device and data-parallel wrapper (if applicable),
        3) algorithm, 4) logger.
        """
        p = psutil.Process()
        try:
            if (self.affinity.get("master_cpus", None) is not None and
                    self.affinity.get("set_affinity", True)):
                p.cpu_affinity(self.affinity["master_cpus"])
            cpu_affin = p.cpu_affinity()
        except AttributeError:
            cpu_affin = "UNAVAILABLE MacOS"
        logger.log(f"Runner {getattr(self, 'rank', '')} master CPU affinity: "
                   f"{cpu_affin}.")

        if self.affinity.get("master_torch_threads", None) is not None:
            torch.set_num_threads(self.affinity["master_torch_threads"])

        logger.log(f"Runner {getattr(self, 'rank', '')} master Torch threads: "
                   f"{torch.get_num_threads()}.")

        if self.seed is None:
            self.seed = make_seed()
        set_seed(self.seed)

        sampler_init_args = dict(
            agent=self.agent,  # Agent gets initialized in sampler.
            affinity=self.affinity,
            seed=self.seed + 1,
            bootstrap_value=getattr(self.algo, "bootstrap_value", False),
            traj_info_kwargs=dict(),
            rank=self.rank,
            world_size=1,
        )
        if isinstance(self.sampler, ParallelSamplerBase):
            sampler_init_args['worker_process'] = ebay_sampling_process
        self.sampler.initialize(**sampler_init_args)

        self.itr_batch_size = self.sampler.batch_spec.size
        self.agent.to_device(self.affinity.get("cuda_idx", None))
        self.algo.initialize(agent=self.agent)
        self.initialize_logging()

    def initialize_logging(self):
        self._start_time = self._last_time = time.time()
        self._cum_time = 0.
        self._cum_completed_trajs = 0
        self._last_update_counter = 0

    def shutdown(self):
        logger.log("Training complete.")
        self.pbar.stop()
        self.sampler.shutdown()

    def get_itr_snapshot(self, itr):
        """
        Returns all state needed for full checkpoint/snapshot of training run,
        including agents parameters and optimizer parameters.
        """
        return dict(itr=itr,
                    cum_steps=itr * self.sampler.batch_size,
                    agent_state_dict=self.agent.state_dict(),
                    optimizer_state_dict=self.algo.optim_state_dict())

    def save_itr_snapshot(self, itr):
        """
        Calls the logger to save training checkpoint/snapshot (logger itself
        may or may not save, depending on mode selected).
        """
        logger.log("saving snapshot...")
        params = self.get_itr_snapshot(itr)
        logger.save_itr_params(itr, params)
        logger.log("saved")

    def store_diagnostics(self, itr, traj_infos, opt_info):
        """
        Store any diagnostic information from a training iteration that should
        be kept for the next logging iteration.
        """
        self._cum_completed_trajs += len(traj_infos)
        for k, v in self._opt_infos.items():
            new_v = getattr(opt_info, k, [])
            v.extend(new_v if isinstance(new_v, list) else [new_v])
        self.pbar.update((itr + 1) % self.log_interval_itrs)

    def log_diagnostics(self, itr, traj_infos=None, eval_time=0, prefix='Diagnostics/'):
        """
        Write diagnostics (including stored ones) to csv via the logger.
        """
        if itr > 0:
            self.pbar.stop()
        if itr >= self.min_itr_learn - 1:
            self.save_itr_snapshot(itr)
        new_time = time.time()
        self._cum_time = new_time - self._start_time
        train_time_elapsed = new_time - self._last_time - eval_time
        new_updates = self.algo.update_counter - self._last_update_counter
        new_samples = (self.sampler.batch_size * self.log_interval_itrs)
        updates_per_second = (new_updates / train_time_elapsed)
        samples_per_second = (new_samples / train_time_elapsed)
        # cum_steps = (itr + 1) * self.sampler.batch_size

        with logger.tabular_prefix(prefix):
            # logger.record_tabular('Iteration', itr)
            # logger.record_tabular('CumSeconds', self._cum_time)
            # logger.record_tabular('CumSteps', cum_steps)
            logger.record_tabular('CumCompletedTrajs', self._cum_completed_trajs)
            # logger.record_tabular('CumUpdates', self.algo.update_counter)
            logger.record_tabular('StepsPerSecond', samples_per_second)
            logger.record_tabular('UpdatesPerSecond', updates_per_second)

        self._log_infos()
        logger.dump_tabular(with_prefix=False)

        self._last_time = new_time
        self._last_update_counter = self.algo.update_counter
        if not self.algo.training_complete:
            logger.log(f"Optimizing over {self.log_interval_itrs} iterations.")
            self.pbar = ProgBarCounter(self.log_interval_itrs)

    def _log_infos(self):
        """
        Writes trajectory info and optimizer info into csv via the logger.
        Resets stored optimizer info.
        """
        if self._opt_infos:
            for k, v in self._opt_infos.items():
                if len(v) == 0:
                    continue
                if type(v[0]) is np.ndarray:
                    logger.record_tabular_misc_stat(k, np.concatenate(v))
                else:
                    k = k.replace('_', '/')
                    logger.record_tabular(k, np.average(v))
        self._opt_infos = {k: list() for k in self._opt_infos}  # (reset)

    def train(self):
        raise NotImplementedError()


class EBayRunner(EBayBaseRunner):
    """
    Runs RL on minibatches; tracks performance online using learning
    trajectories.
    """

    def __init__(self, log_traj_window=100, **kwargs):
        """
        Args:
            log_traj_window (int): How many trajectories to hold in deque for computing performance statistics.
        """
        super().__init__(**kwargs)
        self.log_traj_window = int(log_traj_window)

    def train(self):
        """
        Performs startup, then loops by alternating between
        ``sampler.obtain_samples()`` and ``algo.optimize_agent()``, logging
        diagnostics at the specified interval.
        """
        self.startup()
        itr = 0
        while True:
            logger.set_iteration(itr)
            with logger.prefix(f"itr #{itr} "):
                self.agent.sample_mode(itr)  # Might not be this agents sampling.
                samples, traj_infos = self.sampler.obtain_samples(itr)
                self.agent.train_mode(itr)
                opt_info = self.algo.optimize_agent(samples)
                self.store_diagnostics(itr, traj_infos, opt_info)
                if (itr + 1) % self.log_interval_itrs == 0:
                    self.log_diagnostics(itr)

                if self.algo.training_complete:
                    break
            itr += 1
        self.shutdown()

    def initialize_logging(self):
        self._traj_infos = deque(maxlen=self.log_traj_window)
        self._new_completed_trajs = 0
        logger.log(f"Optimizing over {self.log_interval_itrs} iterations.")
        super().initialize_logging()
        self.pbar = ProgBarCounter(self.log_interval_itrs)

    def store_diagnostics(self, itr, traj_infos, opt_info):
        self._new_completed_trajs += len(traj_infos)
        self._traj_infos.extend(traj_infos)
        super().store_diagnostics(itr, traj_infos, opt_info)

    def log_diagnostics(self, itr, traj_infos=None, eval_time=None, prefix='Diagnostics/'):
        with logger.tabular_prefix(prefix):
            logger.record_tabular('NewCompletedTrajs',
                                  self._new_completed_trajs)
            # steps = sum(info["Length"] for info in self._traj_infos)
            # logger.record_tabular('StepsInTrajWindow', steps)
        super().log_diagnostics(itr, prefix=prefix)
        self._new_completed_trajs = 0


def ebay_sampling_process(common_kwargs, worker_kwargs):
    """Target function used for forking parallel worker processes in the
    samplers. After ``initialize_worker()``, it creates the specified number
    of environment instances and gives them to the collector when
    instantiating it.  It then calls collector startup methods for
    envs and agents.  If applicable, instantiates evaluation
    environment instances and evaluation collector.
    Then enters infinite loop, waiting for signals from master to collect
    training samples or else run evaluation, until signaled to exit.
    """
    c, w = AttrDict(**common_kwargs), AttrDict(**worker_kwargs)
    initialize_worker(w.rank, w.seed, w.cpus, c.torch_threads)
    envs = list()
    x_lstg_cols = c.env_kwargs['composer'].x_lstg_cols
    for env_rank in w.env_ranks:
        loader = TrainLoader(rank=env_rank,
                             x_lstg_cols=x_lstg_cols,
                             byr=c.env_kwargs['composer'].byr)
        envs.append(c.EnvCls(**c.env_kwargs, loader=loader))
    set_envs_seeds(envs, w.seed)
    collector = c.CollectorCls(
        rank=w.rank,
        envs=envs,
        samples_np=w.samples_np,
        batch_T=c.batch_T,
        TrajInfoCls=c.TrajInfoCls,
        agent=c.get("agents", None),  # Optional depending on parallel setup.
        sync=w.get("sync", None),
        step_buffer_np=w.get("step_buffer_np", None),
        global_B=c.get("global_B", 1),
        env_ranks=w.get("env_ranks", None),
    )
    agent_inputs, traj_infos = collector.start_envs(c.max_decorrelation_steps)
    collector.start_agent()

    ctrl = c.ctrl
    ctrl.barrier_out.wait()
    while True:
        collector.reset_if_needed(agent_inputs)  # Outside barrier?
        ctrl.barrier_in.wait()
        if ctrl.quit.value:
            break
        agent_inputs, traj_infos, completed_infos = collector.collect_batch(
            agent_inputs, traj_infos, ctrl.itr.value)
        for info in completed_infos:
            c.traj_infos_queue.put(info)
        ctrl.barrier_out.wait()

    for env in envs:
        env.close()

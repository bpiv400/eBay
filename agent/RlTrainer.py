import os
import psutil
import torch
from datetime import datetime as dt
from agent.algo.SellerPPO import SellerPPO
from agent.algo.BuyerPPO import BuyerPPO
from agent.EBayRunner import EBayMinibatchRl
from agent.models.SplitCategoricalPgAgent import SplitCategoricalPgAgent
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.gpu.alternating_sampler import AlternatingSampler
from rlpyt.utils.logging.context import logger_context
from constants import AGENT_DIR, BYR, SLR
from agent.const import FEAT_TYPE
from agent.AgentComposer import AgentComposer
from agent.models.PgCategoricalAgentModel import PgCategoricalAgentModel
from rlenv.interfaces.PlayerInterface import SimulatedBuyer, SimulatedSeller
from rlenv.interfaces.ArrivalInterface import ArrivalInterface
from rlenv.environments.SellerEnvironment import SellerEnvironment
from rlenv.environments.BuyerEnvironment import BuyerEnvironment
from featnames import BYR_HIST


class RlTrainer:
    def __init__(self, **kwargs):
        # save parameters directly
        self.agent_params = kwargs['agent_params']
        self.econ_params = kwargs['econ_params']
        self.model_params = kwargs['model_params']
        self.ppo_params = kwargs['ppo_params']
        self.system_params = kwargs['system_params']

        # buyer indicator
        self.byr = kwargs['agent_params'][BYR]

        # counts
        self.itr = 0
        self.batch_size = self.system_params['batch_size']

        # initialize composer
        self.composer = AgentComposer(agent_params=self.agent_params)

        # rlpyt components
        self.sampler = self._generate_sampler()
        self.runner = self._generate_runner()

        # for logging
        if self.system_params['log']:
            self.log_dir = self._make_log_dir()
            self.run_id = dt.now().strftime('%y%m%d-%H%M%S')

    def _make_log_dir(self):
        if self.byr:
            log_dir = AGENT_DIR + '{}/hist_{}'.format(
                BYR, self.agent_params[BYR_HIST])
        else:
            log_dir = AGENT_DIR + '{}/{}'.format(
                SLR, self.agent_params[FEAT_TYPE])
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        return log_dir

    def _generate_sampler(self):
        # sampler and batch sizes
        if self.system_params['serial']:
            sampler_cls = SerialSampler
            batch_b = 1
        else:
            sampler_cls = AlternatingSampler
            batch_b = len(self._cpus)

        # environment
        env = BuyerEnvironment if self.byr else SellerEnvironment
        env_params = dict(composer=self.composer,
                          verbose=self.system_params['verbose'],
                          arrival=ArrivalInterface(),
                          seller=SimulatedSeller(full=self.byr),
                          buyer=SimulatedBuyer(full=True))

        return sampler_cls(
                EnvCls=env,
                env_kwargs=env_params,
                batch_B=batch_b,
                batch_T=int(self.batch_size / batch_b),
                max_decorrelation_steps=0,
                eval_n_envs=0,
                eval_env_kwargs={},
                eval_max_steps=50,
            )

    def _generate_algo(self):
        algo = BuyerPPO if self.byr else SellerPPO
        return algo(ppo_params=self.ppo_params,
                    econ_params=self.econ_params)

    def _generate_runner(self):
        agent = SplitCategoricalPgAgent(ModelCls=PgCategoricalAgentModel,
                                        model_kwargs=self.model_params)
        affinity = dict(master_cpus=self._cpus,
                        master_torch_threads=len(self._cpus),
                        workers_cpus=self._cpus,
                        worker_torch_threads=1,
                        cuda_idx=torch.cuda.current_device(),
                        alternating=True,
                        set_affinity=True)
        runner = EBayMinibatchRl(algo=self._generate_algo(),
                                 agent=agent,
                                 sampler=self.sampler,
                                 log_interval_steps=self.batch_size,
                                 affinity=affinity)
        return runner

    @property
    def _cpus(self):
        return list(psutil.Process().cpu_affinity())

    def train(self):
        if not self.system_params['log']:
            self.itr = self.runner.train()
        else:
            with logger_context(log_dir=self.log_dir,
                                name='log',
                                use_summary_writer=True,
                                override_prefix=True,
                                run_ID=self.run_id,
                                snapshot_mode='last'):
                self.itr = self.runner.train()

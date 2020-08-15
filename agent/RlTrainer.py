import psutil
import torch
from agent.EBayRunner import EBayRunner
from agent.EBayPPO import EBayPPO
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.gpu.alternating_sampler import AlternatingSampler
from rlpyt.utils.logging.context import logger_context
from featnames import BYR
from agent.const import AGENT_STATE, BATCH_SIZE
from agent.util import get_paths
from agent.AgentComposer import AgentComposer
from agent.AgentModel import AgentModel
from agent.SplitCategoricalPgAgent import SplitCategoricalPgAgent
from rlenv.DefaultQueryStrategy import DefaultQueryStrategy
from rlenv.environments.SellerEnv import SellerEnv
from rlenv.environments.BuyerEnv import BuyerEnv
from rlenv.interfaces.ArrivalInterface import ArrivalInterface
from rlenv.interfaces.PlayerInterface import SimulatedSeller, SimulatedBuyer
from rlenv.LstgLoader import TrainLoader


class RlTrainer:
    def __init__(self, **params):
        # save params to self
        self.params = params['system']
        self.byr = params[BYR]

        # iteration
        self.itr = 0

        # initialize composer
        self.composer = AgentComposer(byr=self.byr)

        # algorithm
        self.algo = EBayPPO(**params['ppo'])

        # agent
        self.agent = SplitCategoricalPgAgent(
            ModelCls=AgentModel,
            model_kwargs={BYR: self.byr,
                          'serial': self.params['serial']}
        )

        # rlpyt components
        self.sampler = self._generate_sampler()
        self.runner = self._generate_runner()

        # for logging
        self.log_dir, self.run_id, self.run_dir = get_paths(
            byr=self.byr,
            suffix=self.params['suffix'],
            **params['ppo']
        )

    def _generate_query_strategy(self):
        return DefaultQueryStrategy(
            arrival=ArrivalInterface(),
            seller=SimulatedSeller(full=self.byr),
            buyer=SimulatedBuyer(full=True)
        )

    def _generate_sampler(self):
        # environment
        env_params = dict(
            composer=self.composer,
            verbose=self.params['verbose'],
            query_strategy=self._generate_query_strategy()
        )
        env = BuyerEnv if self.byr else SellerEnv

        # sampler and batch sizes
        if self.params['serial']:
            sampler_cls = SerialSampler
            batch_B = 1
            batch_T = 128
            env_params['loader'] = TrainLoader(
                x_lstg_cols=self.composer.x_lstg_cols)
        else:
            sampler_cls = AlternatingSampler
            batch_B = len(self._cpus)
            batch_T = int(BATCH_SIZE / batch_B)

        # environment
        return sampler_cls(
                EnvCls=env,
                env_kwargs=env_params,
                batch_B=batch_B,
                batch_T=batch_T,
                max_decorrelation_steps=0,
                eval_n_envs=0,
                eval_env_kwargs={},
                eval_max_steps=50,
            )

    def _generate_runner(self):
        affinity = dict(master_cpus=self._cpus,
                        master_torch_threads=len(self._cpus),
                        workers_cpus=self._cpus,
                        worker_torch_threads=1,
                        cuda_idx=torch.cuda.current_device(),
                        alternating=True,
                        set_affinity=True)
        runner = EBayRunner(algo=self.algo,
                            agent=self.agent,
                            sampler=self.sampler,
                            affinity=affinity)
        return runner

    @property
    def _cpus(self):
        return list(psutil.Process().cpu_affinity())

    def train(self):
        if not self.params['log']:
            self.itr = self.runner.train()
        else:
            with logger_context(log_dir=self.log_dir,
                                name='log',
                                use_summary_writer=True,
                                override_prefix=True,
                                run_ID=self.run_id,
                                snapshot_mode='last'):
                self.itr = self.runner.train()

            # delete optimization parameters
            path = self.run_dir + 'params.pkl'
            d = torch.load(path)
            torch.save(d[AGENT_STATE], path)

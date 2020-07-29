import psutil
import torch
from agent.EBayRunner import EBayMinibatchRl
from agent.EBayPPO import EBayPPO
from agent.models.SplitCategoricalPgAgent import SplitCategoricalPgAgent
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.gpu.alternating_sampler import AlternatingSampler
from rlpyt.utils.logging.context import logger_context
from constants import BYR
from agent.const import AGENT_STATE
from agent.util import get_log_dir, get_run_id
from agent.AgentComposer import AgentComposer
from agent.models.PgCategoricalAgentModel import PgCategoricalAgentModel
from rlenv.DefaultQueryStrategy import DefaultQueryStrategy
from rlenv.environments.SellerEnvironment import SellerEnvironment
from rlenv.environments.BuyerEnvironment import BuyerEnvironment
from rlenv.interfaces.ArrivalInterface import ArrivalInterface
from rlenv.interfaces.PlayerInterface import SimulatedSeller, SimulatedBuyer
from rlenv.LstgLoader import TrainLoader


class RlTrainer:
    def __init__(self, **params):
        # save params to self
        self.system_params = params['system']
        self.model_params = params['model']

        # buyer indicator
        self.byr = self.model_params[BYR]

        # counts
        self.itr = 0
        self.batch_size = self.system_params['batch_size']

        # initialize composer
        self.composer = AgentComposer(byr=self.byr)

        # algorithm
        self.algo = EBayPPO(byr=self.byr, **params['econ'])

        # agent
        self.agent = SplitCategoricalPgAgent(
            ModelCls=PgCategoricalAgentModel,
            model_kwargs=self.model_params
        )

        # rlpyt components
        self.sampler = self._generate_sampler()
        self.runner = self._generate_runner()

        # for logging
        self.log_dir = get_log_dir(byr=self.byr)
        self.run_id = get_run_id(**params['econ'])
        self.run_dir = self.log_dir + 'run_{}/'.format(self.run_id)

    def _generate_query_strategy(self):
        return DefaultQueryStrategy(
            arrival=ArrivalInterface(),
            seller=SimulatedSeller(full=self.byr),
            buyer=SimulatedBuyer(full=True)
        )

    def _generate_sampler(self):
        env_params = dict(composer=self.composer,
                          verbose=self.system_params['verbose'],
                          query_strategy=self._generate_query_strategy(),
                          recorder=None)
        # sampler and batch sizes
        if self.system_params['serial']:
            sampler_cls = SerialSampler
            batch_b = 1
            x_lstg_cols = env_params['composer'].x_lstg_cols
            env_params['loader'] = TrainLoader(x_lstg_cols=x_lstg_cols)
        else:
            sampler_cls = AlternatingSampler
            batch_b = len(self._cpus)

        # environment
        env = BuyerEnvironment if self.byr else SellerEnvironment
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

    def _generate_runner(self):
        affinity = dict(master_cpus=self._cpus,
                        master_torch_threads=len(self._cpus),
                        workers_cpus=self._cpus,
                        worker_torch_threads=1,
                        cuda_idx=torch.cuda.current_device(),
                        alternating=True,
                        set_affinity=True)
        runner = EBayMinibatchRl(algo=self.algo,
                                 agent=self.agent,
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

            # delete optimization parameters
            path = self.run_dir + 'params.pkl'
            d = torch.load(path)
            torch.save(d[AGENT_STATE], path)

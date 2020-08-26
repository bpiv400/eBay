import psutil
import torch
from agent.EBayRunner import EBayRunner
from agent.EBayPPO import EBayPPO
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.gpu.alternating_sampler import AlternatingSampler
from rlpyt.utils.logging.context import logger_context
from agent.const import AGENT_STATE, BATCH_SIZE, ENTROPY
from agent.util import get_paths
from agent.AgentComposer import AgentComposer
from agent.models.AgentModel import AgentModel, SplitCategoricalPgAgent
from rlenv.QueryStrategy import DefaultQueryStrategy
from agent.envs.SellerEnv import SellerEnv
from agent.envs.BuyerEnv import BuyerEnv
from rlenv.interfaces.ArrivalInterface import ArrivalInterface
from rlenv.interfaces.PlayerInterface import SimulatedSeller, SimulatedBuyer
from rlenv.LstgLoader import TrainLoader


class RlTrainer:
    def __init__(self, byr=False, serial=False, name=None, con_set=False):
        # save params to self
        self.byr = byr
        self.con_set = con_set

        # iteration
        self.itr = 0

        # initialize composer
        self.composer = AgentComposer(byr=byr)

        # algorithm
        entropy = ENTROPY[con_set]
        self.algo = EBayPPO(entropy=entropy)

        # agent
        self.agent = SplitCategoricalPgAgent(
            ModelCls=AgentModel,
            model_kwargs=dict(byr=byr, serial=serial, con_set=con_set)
        )

        # rlpyt components
        self.sampler = self._generate_sampler(serial=serial)
        self.runner = self._generate_runner()

        # for logging
        self.log_dir, self.run_id, self.run_dir = \
            get_paths(byr=byr, name=name)

    def _generate_query_strategy(self):
        return DefaultQueryStrategy(
            arrival=ArrivalInterface(),
            seller=SimulatedSeller(full=self.byr),
            buyer=SimulatedBuyer(full=True)
        )

    def _generate_sampler(self, serial=False):
        # environment
        env_params = dict(
            composer=self.composer,
            verbose=serial,
            query_strategy=self._generate_query_strategy(),
            con_set=self.con_set
        )
        env = BuyerEnv if self.byr else SellerEnv

        # sampler and batch sizes
        if serial:
            sampler_cls = SerialSampler
            batch_B = 1
            batch_T = 128
            env_params['loader'] = TrainLoader(
                x_lstg_cols=self.composer.x_lstg_cols,
                byr=self.byr
            )
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

    def train(self, log=False):
        if not log:
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

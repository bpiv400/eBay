import psutil
import torch
from agent.EBayRunner import EBayRunner
from agent.EBayPPO import EBayPPO
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.gpu.alternating_sampler import AlternatingSampler
from rlpyt.utils.logging.context import logger_context
from agent.util import get_run_id
from agent.AgentComposer import AgentComposer
from agent.models.AgentModel import AgentModel
from agent.agents import SellerAgent, BuyerAgent
from rlenv.QueryStrategy import DefaultQueryStrategy
from agent.envs.SellerEnv import SellerEnv
from agent.envs.BuyerEnv import BuyerEnv
from rlenv.interfaces.ArrivalInterface import ArrivalInterface
from rlenv.interfaces.PlayerInterface import SimulatedSeller, SimulatedBuyer
from agent.AgentLoader import AgentLoader
from agent.const import BATCH_SIZE
from constants import AGENT_DIR


class RlTrainer:
    def __init__(self, delta=None):
        self.delta = delta
        self.byr = delta is None

    def _generate_query_strategy(self):
        return DefaultQueryStrategy(
            arrival=ArrivalInterface(),
            seller=SimulatedSeller(full=self.byr),
            buyer=SimulatedBuyer(full=True)
        )

    def _generate_agent(self, serial=False):
        model_kwargs = dict(byr=self.byr)
        agent_cls = BuyerAgent if self.byr else SellerAgent
        return agent_cls(
            ModelCls=AgentModel,
            model_kwargs=model_kwargs,
            serial=serial
        )

    def _generate_env(self, verbose=False):
        composer = AgentComposer(byr=self.byr)
        env_params = dict(
            composer=composer,
            verbose=verbose,
            query_strategy=self._generate_query_strategy(),
            loader=AgentLoader(),
            delta=self.delta,
            train=True
        )
        env = BuyerEnv if self.byr else SellerEnv
        return env, env_params

    def _generate_sampler(self, serial=False):
        env, env_params = self._generate_env(serial)

        # sampler and batch sizes
        if serial:
            sampler_cls = SerialSampler
            batch_B = 1
            batch_T = 128
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

    def _generate_runner(self, serial=None):
        affinity = dict(master_cpus=self._cpus,
                        master_torch_threads=len(self._cpus),
                        workers_cpus=self._cpus,
                        worker_torch_threads=1,
                        cuda_idx=torch.cuda.current_device(),
                        alternating=True,
                        set_affinity=True)
        return EBayRunner(algo=EBayPPO(),
                          agent=self._generate_agent(serial),
                          sampler=self._generate_sampler(serial),
                          affinity=affinity)

    @property
    def _cpus(self):
        return list(psutil.Process().cpu_affinity())

    def train(self, log=None, serial=None):
        # construct runner
        runner = self._generate_runner(serial)

        if not log:
            runner.train()
        else:
            run_id = get_run_id(delta=self.delta)
            with logger_context(log_dir=AGENT_DIR,
                                name='log',
                                use_summary_writer=True,
                                override_prefix=True,
                                run_ID=run_id,
                                snapshot_mode='last'):
                runner.train()

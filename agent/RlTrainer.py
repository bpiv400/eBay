import psutil
import os
import torch
from agent.EBayRunner import EBayRunner
from agent.EBayPPO import EBayPPO
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.gpu.alternating_sampler import AlternatingSampler
from rlpyt.utils.logging.context import logger_context
from agent.const import AGENT_STATE, BATCH_SIZE
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
    def __init__(self, byr=False, con_set=False):
        self.byr = byr
        self.con_set = con_set

    def _generate_query_strategy(self):
        return DefaultQueryStrategy(
            arrival=ArrivalInterface(),
            seller=SimulatedSeller(full=self.byr),
            buyer=SimulatedBuyer(full=True)
        )

    def _generate_agent(self, serial=False):
        kwargs = dict(
                byr=self.byr,
                serial=serial,
                con_set=self.con_set
            )
        return SplitCategoricalPgAgent(
            ModelCls=AgentModel,
            model_kwargs=kwargs
        )

    def _generate_sampler(self, serial=False):
        # environment
        composer = AgentComposer(byr=self.byr)
        env_params = dict(
            composer=composer,
            verbose=serial,
            query_strategy=self._generate_query_strategy(),
            con_set=self.con_set,
            loader=TrainLoader(
                x_lstg_cols=composer.x_lstg_cols,
                byr=self.byr
            )
        )
        env = BuyerEnv if self.byr else SellerEnv

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

    def _generate_runner(self, agent=None, sampler=None):
        algo = EBayPPO()
        affinity = dict(master_cpus=self._cpus,
                        master_torch_threads=len(self._cpus),
                        workers_cpus=self._cpus,
                        worker_torch_threads=1,
                        cuda_idx=torch.cuda.current_device(),
                        alternating=True,
                        set_affinity=True)
        return EBayRunner(algo=algo,
                          agent=agent,
                          sampler=sampler,
                          affinity=affinity)

    @property
    def _cpus(self):
        return list(psutil.Process().cpu_affinity())

    def train(self, **kwargs):
        if kwargs['serial']:
            assert not kwargs['log']

        # construct runner
        agent = self._generate_agent(kwargs['serial'])
        sampler = self._generate_sampler(kwargs['serial'])
        runner = self._generate_runner(agent=agent, sampler=sampler)

        if not kwargs['log']:
            runner.train()
        else:
            log_dir, run_id, run_dir = get_paths(byr=self.byr,
                                                 con_set=self.con_set,
                                                 suffix=kwargs['suffix'])
            if os.path.isdir(run_dir):
                print('{} already exists.'.format(run_id))
                exit()

            with logger_context(log_dir=log_dir,
                                name='log',
                                use_summary_writer=True,
                                override_prefix=True,
                                run_ID=run_id,
                                snapshot_mode='last'):
                runner.train()

            # delete optimization parameters
            path = run_dir + 'params.pkl'
            d = torch.load(path)
            torch.save(d[AGENT_STATE], path)

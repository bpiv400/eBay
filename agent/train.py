"""
Train a seller agent that makes concessions, not offers
"""
import argparse
from datetime import datetime as dt
import multiprocessing as mp
import warnings
import torch
from agent.CrossEntropyPPO import CrossEntropyPPO
from agent.EBayRunner import EBayMinibatchRl
from agent.models.SplitCategoricalPgAgent import SplitCategoricalPgAgent
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.utils.logging.context import logger_context
from featnames import DELAY
from constants import RL_LOG_DIR, BYR_PREFIX, PARTS_DIR, TRAIN_RL, DROPOUT
from agent.agent_consts import (AGENT_STATE, PARAM_DICTS, THREADS_PER_PROC)
from agent.agent_utils import gen_run_id, save_params
from agent.AgentComposer import AgentComposer
from agent.models.PgCategoricalAgentModel import PgCategoricalAgentModel
from rlenv.env_utils import load_chunk
from rlenv.interfaces.PlayerInterface import SimulatedBuyer, SimulatedSeller
from rlenv.interfaces.ArrivalInterface import ArrivalInterface
from rlenv.environments.SellerEnvironment import SellerEnvironment


class RlTrainer:
    def __init__(self, **kwargs):
        self.agent_params = kwargs['agent_params']
        self.batch_params = kwargs['batch_params']
        self.ppo_params = kwargs['ppo_params']
        self.system_params = kwargs['system_params']

        # iteration
        self.itr = 0

        # ids
        self.run_id = gen_run_id()
        self.log_dir = RL_LOG_DIR + '{}/'.format(self.agent_params['role'])

        # parameters
        self.env_params_train = self.generate_train_params()

        # rlpyt components
        self.sampler = self.generate_sampler()
        self.runner = self.generate_runner()

    def generate_train_params(self):
        chunk_path = PARTS_DIR + '{}/chunks/1.gz'.format(TRAIN_RL)
        x_lstg_cols = load_chunk(input_path=chunk_path)[0].columns
        composer = AgentComposer(cols=x_lstg_cols,
                                 agent_params=self.agent_params)
        env_params = {
            'composer': composer,
            'verbose': self.system_params['verbose'],
            'arrival': ArrivalInterface(),
            'seller': SimulatedSeller(full=False),
            'buyer': SimulatedBuyer()
        }
        return env_params

    def generate_algorithm(self):
        return CrossEntropyPPO(**self.ppo_params)

    def generate_agent(self):
        # initialize model keyword arguments
        model_kwargs = dict()
        model_kwargs[BYR_PREFIX] = self.agent_params['role'] == BYR_PREFIX
        model_kwargs[DELAY] = self.agent_params[DELAY]
        model_kwargs[DROPOUT] = (self.agent_params['dropout0'],
                                 self.agent_params['dropout1'])

        return SplitCategoricalPgAgent(ModelCls=PgCategoricalAgentModel,
                                       model_kwargs=model_kwargs)

    def generate_sampler(self):
        batch_b = len(self.workers_cpus) * 2
        batch_t = int(self.batch_params['batch_size'] / batch_b)
        if batch_t < 12:
            warnings.warn("Very few actions per environment")
        if self.system_params['debug']:
            return SerialSampler(
                EnvCls=SellerEnvironment,
                env_kwargs=self.env_params_train,
                batch_B=batch_b,
                batch_T=batch_t,
                max_decorrelation_steps=0,
                eval_n_envs=0,
                eval_env_kwargs={},
                eval_max_steps=50,
            )
        else:
            if self.system_params['gpu']:
                sampler_class = GpuSampler
            else:
                sampler_class = CpuSampler
            return sampler_class(
                EnvCls=SellerEnvironment,
                env_kwargs=self.env_params_train,
                batch_B=batch_b,
                batch_T=batch_t,
                max_decorrelation_steps=0,
                eval_n_envs=0,
                eval_env_kwargs={},
                eval_max_steps=50,
            )

    def generate_runner(self):
        runner = EBayMinibatchRl(algo=self.generate_algorithm(),
                                 agent=self.generate_agent(),
                                 sampler=self.sampler,
                                 log_interval_steps=self.batch_params['batch_size'],
                                 affinity=self.generate_affinity())
        return runner

    def generate_affinity(self):
        affinity = dict(workers_cpus=self.workers_cpus,
                        master_torch_threads=THREADS_PER_PROC,
                        cuda_idx=0,
                        set_affinity=not self.system_params['auto'])
        return affinity

    @property
    def workers_cpus(self):
        if not self.system_params['auto']:
            cpus = self.workers_cpus_manual()
        else:
            cpus = self.eligible_cpus[:self.system_params['workers']]
        return cpus

    def workers_cpus_manual(self):
        eligible = self.eligible_cpus
        worker_count = self.system_params['workers']
        if self.system_params['multiple']:
            threads_per_worker = int(len(eligible) / worker_count)
            cpus = list()
            for i in range(worker_count):
                curr = eligible[(i * threads_per_worker): (i+1) * threads_per_worker]
                cpus.append(curr)
            for j in range((len(eligible) // worker_count)):
                cpus[j].append(eligible[j + threads_per_worker * worker_count])
        else:
            cpus = eligible[:worker_count]
        return cpus

    def train(self):
        with logger_context(log_dir=self.log_dir,
                            name='debug',
                            use_summary_writer=True,
                            override_prefix=True,
                            run_ID=self.run_id,
                            snapshot_mode='last'):
            self.itr = self.runner.train()
    
    @property
    def eligible_cpus(self):
        workers_cpu = list(range(mp.cpu_count()))
        if len(workers_cpu) == 64:
            workers_cpu.remove(33)
            workers_cpu.remove(1)
        elif len(workers_cpu) == 32:
            workers_cpu.remove(1)
            workers_cpu.remove(17)
        return workers_cpu


def main():
    parser = argparse.ArgumentParser()
    # experiment parameters
    for d in PARAM_DICTS.values():
        for k, v in d.items():
            parser.add_argument('--{}'.format(k), **v)
    args = vars(parser.parse_args())
    for k, v in args.items():
        print('{}: {}'.format(k, v))

    # split parameters
    trainer_args = dict()
    for param_set, param_dict in PARAM_DICTS.items():
        curr_params = dict()
        for k in param_dict.keys():
            curr_params[k] = args[k]
        trainer_args[param_set] = curr_params

    # initialize trainer
    trainer = RlTrainer(**trainer_args)

    # training loop
    t0 = dt.now()
    trainer.train()
    time_elapsed = (dt.now() - t0).total_seconds()

    # save parameters to file
    save_params(run_id=trainer.run_id,
                args=args,
                time_elapsed=time_elapsed)

    # drop optimization parameters
    path = trainer.log_dir + 'run_{}/params.pkl'.format(trainer.run_id)
    d = torch.load(path)
    torch.save(d[AGENT_STATE], path)


if __name__ == '__main__':
    main()

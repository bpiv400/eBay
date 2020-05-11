"""
Train a seller agent that makes concessions, not offers
"""
import os
import argparse
from datetime import datetime as dt
import multiprocessing as mp
import warnings
import torch
from agent.CrossEntropyPPO import CrossEntropyPPO
from agent.EBayRunner import EBayMinibatchRl
from agent.SplitCategoricalPgAgent import SplitCategoricalPgAgent
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

# remember to deprecate these
WORKERS = 12
ASSIGN_CPUS = True
MULTIPLE_CPUS = False


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

        # minibatches in ppo_params
        self.ppo_params['minibatches'] = \
            int(self.batch_params['batch_size'] / self.ppo_params['mbsize'])
        del self.ppo_params['mbsize']

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
        model_kwargs[DROPOUT] = tuple(self.agent_params[DROPOUT])

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
        if ASSIGN_CPUS:
            workers_cpus = self.workers_cpus
            n_worker = None
        else:
            workers_cpus = None
            n_worker = len(self.workers_cpus)

        affinity = dict(workers_cpus=workers_cpus,
                        n_worker=n_worker,
                        master_torch_threads=THREADS_PER_PROC,
                        cuda_idx=0,
                        set_affinity=ASSIGN_CPUS)
        return affinity

    @property
    def workers_cpus(self):
        if ASSIGN_CPUS:
            cpus = self.workers_cpus_manual()
        else:
            cpus = list(range(WORKERS))
        return cpus

    @staticmethod
    def workers_cpus_manual():
        eligible = list(range(mp.cpu_count()))
        if mp.cpu_count() == 64:
            eligible.remove(1)
            eligible.remove(33)
        if MULTIPLE_CPUS:
            threads_per_worker = int(len(eligible) / WORKERS)
            cpus = list()
            for i in range(WORKERS):
                curr = eligible[(i * threads_per_worker): (i+1) * threads_per_worker]
                cpus.append(curr)
            for j in range((len(eligible) // WORKERS)):
                cpus[j].append(eligible[j + threads_per_worker * WORKERS])
        else:
            cpus = eligible[0:WORKERS]
        return cpus

    def train(self):
        with logger_context(log_dir=self.log_dir,
                            name='debug',
                            use_summary_writer=True,
                            override_prefix=True,
                            run_ID=self.run_id,
                            snapshot_mode='all'):
            self.itr = self.runner.train()
    
    @property
    def worker_cpus(self):
        workers_cpu = list(range(mp.cpu_count()))
        if len(workers_cpu) == 64:
            workers_cpu.remove(33)
            workers_cpu.remove(1)
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
    save_params(role=agent_params['role'],
                run_id=trainer.run_id,
                agent_params=agent_params,
                batch_params=batch_params,
                ppo_params=ppo_params,
                time_elapsed=time_elapsed)

    # create new subfolders
    run_dir = trainer.log_dir + 'run_{}/'.format(trainer.run_id)
    for name in ['models', 'rewards', 'outcomes']:
        os.mkdir(run_dir + '{}/'.format(name))

    # drop optimization parameters
    for i in range(batch_params['batch_count']):
        # load params
        in_path = run_dir + 'itr_{}.pkl'.format(i)
        d = torch.load(in_path)
        # save model
        out_path = run_dir + 'models/{}.net'.format(i)
        torch.save(d[AGENT_STATE], out_path)
        # delete params
        os.remove(in_path)


if __name__ == '__main__':
    main()

"""
Train a seller agent that makes concessions, not offers
"""
import os
import argparse
from datetime import datetime as dt
import multiprocessing as mp
import torch
from agent.CrossEntropyPPO import CrossEntropyPPO
from rlpyt.algos.pg.ppo import PPO
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector
from rlpyt.samplers.serial.sampler import SerialEvalCollector
from rlpyt.samplers.parallel.cpu.collectors import CpuEvalCollector
from rlpyt.utils.logging.context import logger_context
from featnames import DELAY
from constants import RL_EVAL_DIR, RL_LOG_DIR, BYR_PREFIX
from agent.agent_consts import (SELLER_TRAIN_INPUT, AGENT_STATE,
                                PARAM_DICTS, AGENT_PARAMS,
                                BATCH_PARAMS, PPO_PARAMS,
                                THREADS_PER_PROC)
from agent.agent_utils import gen_run_id, save_params, generate_model_kwargs
from agent.AgentComposer import AgentComposer
from agent.models.PgCategoricalAgentModel import PgCategoricalAgentModel
from rlenv.env_utils import load_chunk
from rlenv.interfaces.PlayerInterface import SimulatedBuyer, SimulatedSeller
from rlenv.interfaces.ArrivalInterface import ArrivalInterface
from rlenv.environments.SellerEnvironment import SellerEnvironment


class RlTrainer:
    def __init__(self, **kwargs):
        # arguments
        self.debug = kwargs['debug']
        self.verbose = kwargs['verbose']
        self.agent_params = kwargs['agent_params']
        self.batch_params = kwargs['batch_params']
        self.ppo_params = kwargs['ppo_params']

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
        chunk_path = '{}1.gz'.format(RL_EVAL_DIR)
        x_lstg_cols = load_chunk(input_path=chunk_path)[0].columns
        composer = AgentComposer(cols=x_lstg_cols,
                                 agent_params=self.agent_params)
        env_params = {
            'composer': composer,
            'verbose': self.verbose,
            'filename': SELLER_TRAIN_INPUT,
            'arrival': ArrivalInterface(),
            'seller': SimulatedSeller(full=False),
            'buyer': SimulatedBuyer()
        }
        return env_params

    def generate_algorithm(self):
        return CrossEntropyPPO(**self.ppo_params)
        # return PPO(**self.ppo_params)

    def generate_agent(self):
        sizes = self.env_params_train['composer'].agent_sizes
        byr = self.agent_params['role'] == BYR_PREFIX
        delay = self.agent_params[DELAY]
        model_kwargs = generate_model_kwargs(sizes, byr, delay)
        return CategoricalPgAgent(ModelCls=PgCategoricalAgentModel,
                                  model_kwargs=model_kwargs)

    def generate_sampler(self):
        batch_b = mp.cpu_count() * 2
        batch_t = int(self.batch_params['batch_size'] / batch_b)
        if self.debug:
            return SerialSampler(
                EnvCls=SellerEnvironment,
                env_kwargs=self.env_params_train,
                batch_B=batch_b,
                batch_T=batch_t,
                max_decorrelation_steps=0,
                CollectorCls=CpuResetCollector,
                eval_n_envs=0,
                eval_CollectorCls=SerialEvalCollector,
                eval_env_kwargs={},
                eval_max_steps=50,
            )
        else:
            return CpuSampler(
                EnvCls=SellerEnvironment,
                env_kwargs=self.env_params_train,
                batch_B=batch_b,
                batch_T=batch_t,
                max_decorrelation_steps=0,
                CollectorCls=CpuResetCollector,
                eval_n_envs=0,
                eval_CollectorCls=CpuEvalCollector,
                eval_env_kwargs={},
                eval_max_steps=50,
            )

    def generate_runner(self):
        workers_cpu = list(range(mp.cpu_count()))
        affinity = dict(workers_cpus=workers_cpu,
                        master_torch_threads=THREADS_PER_PROC,
                        cuda_idx=0)
        runner = MinibatchRl(algo=self.generate_algorithm(),
                             agent=self.generate_agent(),
                             sampler=self.sampler,
                             n_steps=self.batch_params['batch_size'] * self.batch_params['batch_count'],
                             log_interval_steps=self.batch_params['batch_size'],
                             affinity=affinity)
        return runner

    def train(self):
        with logger_context(log_dir=self.log_dir,
                            name='debug',
                            use_summary_writer=True,
                            override_prefix=True,
                            run_ID=self.run_id,
                            snapshot_mode='all'):
            self.itr = self.runner.train()


def main():
    parser = argparse.ArgumentParser()
    # basic arguments
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    # experiment parameters
    for d in PARAM_DICTS:
        for k, v in d.items():
            parser.add_argument('--{}'.format(k), **v)
    args = vars(parser.parse_args())
    for k, v in args.items():
        print('{}: {}'.format(k, v))

    # split parameters
    agent_params, ppo_params, batch_params = dict(), dict(), dict()
    for k in AGENT_PARAMS.keys():
        agent_params[k] = args[k]
    for k in BATCH_PARAMS.keys():
        batch_params[k] = args[k]
    for k in PPO_PARAMS.keys():
        ppo_params[k] = args[k]

    # initialize trainer
    trainer = RlTrainer(debug=args['debug'],
                        verbose=args['verbose'],
                        agent_params=agent_params,
                        batch_params=batch_params,
                        ppo_params=ppo_params)

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

    # drop optimization parameters
    run_dir = trainer.log_dir + 'run_{}/'.format(trainer.run_id)
    for i in range(batch_params['batch_count']):
        # load params
        in_path = run_dir + 'itr_{}.pkl'.format(i)
        d = torch.load(in_path)
        # save model
        itr_dir = run_dir + 'itr/{}/'.format(i)
        if not os.path.isdir(itr_dir):
            os.makedirs(itr_dir)
        torch.save(d[AGENT_STATE], itr_dir + 'agent.net')
        # delete params
        os.remove(in_path)


if __name__ == '__main__':
    main()

"""
Train a seller agent that makes concessions, not offers
"""
import argparse
from datetime import datetime as dt
import multiprocessing as mp
from torch.utils.tensorboard.writer import SummaryWriter
from rlpyt.algos.pg.ppo import PPO
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector
from rlpyt.samplers.serial.sampler import SerialEvalCollector
from rlpyt.samplers.parallel.cpu.collectors import CpuEvalCollector
from rlpyt.utils.logging import logger
from rlpyt.utils.logging.context import logger_context
from featnames import DELAY
from constants import (RL_EVAL_DIR, RL_LOG_DIR, SLR_INIT, BYR_INIT,
                       BYR_PREFIX)
from agent.agent_consts import (SELLER_TRAIN_INPUT, PARAM_DICTS,
                                AGENT_PARAMS, BATCH_PARAMS, PPO_PARAMS)
from agent.agent_utils import load_init_model, detect_norm, \
    gen_run_id, save_params
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

        # fields
        self.itr = 0
        self.norm = None

        # ids
        self.run_id = gen_run_id()
        self.log_dir = RL_LOG_DIR + '{}/'.format(self.agent_params['role'])
        self.run_dir = '{}{}/'.format(self.log_dir, self.run_id)

        # parameters
        self.env_params_train = self.generate_train_params()

        # rlpyt components
        self.sampler = self.generate_sampler()
        self.runner = self.generate_runner()

        # logging setup
        self.writer = SummaryWriter(self.run_dir)

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
        return PPO(**self.ppo_params)

    def generate_model_kwargs(self):
        model_kwargs = {
            'sizes': self.env_params_train['composer'].agent_sizes,
            BYR_PREFIX: self.agent_params['role'] == BYR_PREFIX,
            DELAY: self.agent_params[DELAY]
        }
        # load simulator model to initialize policy
        if self.itr == 0:
            if not model_kwargs[BYR_PREFIX]:
                init_model = SLR_INIT
            else:
                init_model = BYR_INIT
            init_dict = load_init_model(name=init_model,
                                        size=model_kwargs['sizes']['out'])
            self.norm = detect_norm(init_dict)
            model_kwargs['init_dict'] = init_dict
        # set norm type
        model_kwargs['norm'] = self.norm
        return model_kwargs

    def generate_agent(self):
        return CategoricalPgAgent(ModelCls=PgCategoricalAgentModel,
                                  model_kwargs=self.generate_model_kwargs())

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
        affinity = dict(workers_cpus=list(range(mp.cpu_count())),
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
                            use_summary_writer=False,
                            override_prefix=True,
                            run_ID=self.run_id,
                            snapshot_mode='last'):
            logger.set_tf_summary_writer(self.writer)
            self.itr = self.runner.train()
            self.writer.flush()


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


if __name__ == '__main__':
    main()

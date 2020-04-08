"""
Train a seller agent that makes concessions, not offers
"""
import argparse
import os
import time
import re
import shutil
import multiprocessing as mp
import queue
from os.path import isfile, join
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from agent.EvalGenerator import EvalGenerator
from rlpyt.algos.pg.ppo import PPO
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector
from rlpyt.samplers.serial.sampler import SerialEvalCollector
from rlpyt.samplers.parallel.cpu.collectors import CpuEvalCollector
from rlpyt.utils.logging import logger
from rlpyt.utils.logging.context import logger_context
from featnames import DELAY
from constants import (RL_EVAL_DIR, RL_LOG_DIR, SLR_INIT, BYR_INIT,
                       BYR_PREFIX, SLR_PREFIX)
from agent.agent_consts import (CON_TYPE, ALL_FEATS, AGENT_STATE, OPTIM_STATE,
                                SELLER_TRAIN_INPUT, FEAT_TYPE, INIT_LR,
                                QUARTILES)
from agent.agent_utils import load_init_model, detect_norm
from agent.AgentComposer import AgentComposer
from agent.models.PgCategoricalAgentModel import PgCategoricalAgentModel
from agent.runners.EbayRunner import EbayRunner
from rlenv.env_utils import load_chunk
from rlenv.interfaces.PlayerInterface import SimulatedBuyer, SimulatedSeller
from rlenv.interfaces.ArrivalInterface import ArrivalInterface
from rlenv.environments.SellerEnvironment import SellerEnvironment
from train.train_consts import FTOL


class RlTrainer:
    def __init__(self, **kwargs):
        # arguments
        self.debug = kwargs['debug']
        self.verbose = kwargs['verbose']
        self.agent_params = kwargs['agent_params']

        # hyper params
        self.batch_size = kwargs['batch_size']
        self.batches_per_evaluation = kwargs['batches_per_evaluation']
        self.ppo_minibatches = kwargs['ppo_minibatches']
        self.ppo_epochs = kwargs['ppo_epochs']

        # fields
        self.itr = 0
        self.lr = INIT_LR
        self.eval_scores = list()
        self.norm = None
        self.evaluation_chunks = self.count_eval_chunks()
        self.cpu_count = self.count_cpus()
        self.checkpoint = self.init_checkpoint()

        # ids
        self.run_id = self.generate_run_id()  # TODO: Make util function
        self.run_dir = '{}/run_{}/'.format(self.log_dir, self.run_id)

        # parameters
        self.env_params_train = self.generate_train_params()

        # rlpyt components
        self.sampler = self.generate_sampler()
        self.runner = self.generate_runner()

        # logging setup
        self.clear_log()
        self.writer = SummaryWriter(self.run_dir)

    @staticmethod
    def init_checkpoint():
        return {
             AGENT_STATE: None,
             OPTIM_STATE: None
        }

    def generate_run_id(self):
        return "bs-{}_bpe-{}_mb-{}_epochs-{}".format(self.batch_size,
                                                     self.batches_per_evaluation,
                                                     self.ppo_minibatches,
                                                     self.ppo_epochs)

    @staticmethod
    def count_cpus():
        return int(mp.cpu_count() / 2)

    def clear_log(self):
        if os.path.exists(self.run_dir):
            shutil.rmtree(self.run_dir, ignore_errors=True)

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
        return PPO(minibatches=self.ppo_minibatches,
                   epochs=self.ppo_epochs,
                   learning_rate=self.lr,
                   linear_lr_schedule=False,
                   initial_optim_state_dict=self.checkpoint[OPTIM_STATE])

    def generate_model_kwargs(self):
        model_kwargs = {
            'sizes': self.env_params_train['composer'].agent_sizes,
            'byr': self.agent_params[BYR_PREFIX],
            'delay': self.agent_params[DELAY]
        }
        # load simulator model to initialize policy
        if self.itr == 0:
            if not self.agent_params[BYR_PREFIX]:
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
                                  model_kwargs=self.generate_model_kwargs(),
                                  initial_model_state_dict=self.checkpoint[AGENT_STATE])

    def generate_sampler(self):
        batch_b = self.cpu_count * 2
        batch_t = int(self.batch_size / batch_b)
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
        affinity = dict(workers_cpus=list(range(self.cpu_count)))
        runner = EbayRunner(algo=self.generate_algorithm(),
                            agent=self.generate_agent(),
                            sampler=self.sampler,
                            batches_per_evaluation=self.batches_per_evaluation,
                            batch_size=self.batch_size,
                            affinity=affinity)
        return runner

    @property
    def log_dir(self):
        extension = BYR_PREFIX if self.agent_params[BYR_PREFIX] else SLR_PREFIX
        return '{}{}/'.format(RL_LOG_DIR, extension)

    def train(self):
        with logger_context(log_dir=self.log_dir, name='debug', use_summary_writer=False,
                            override_prefix=True, run_ID=self.run_id,
                            snapshot_mode='last'):
            logger.set_tf_summary_writer(self.writer)
            for i in range(10):
                self._train_iteration()

    def _train_iteration(self):
        eval_score = self.evaluate()
        if self.lr_update_needed(last_eval=eval_score):
            self.update_lr()
        if self.itr != 0:
            self.runner.update_agent(self.generate_agent())
            self.runner.update_algo(self.generate_algorithm())
        self.itr = self.runner.train()
        self.writer.flush()
        self.update_checkpoint()

    def evaluate(self):
        start_time = time.time()
        if self.debug:
            rewards = self._serial_evaluate()
        else:
            rewards = self._parallel_evaluate()
        end_time = time.time()
        self._log_eval(rewards=rewards, eval_time=end_time - start_time)
        return np.mean(rewards)

    def _log_eval(self, rewards=None, eval_time=None):
        logger.set_iteration(self.itr)
        logger.record_tabular('evalTime', eval_time)
        logger.record_tabular_misc_stat('evalReward', rewards)

    def _serial_evaluate(self):
        rewards = list()
        eval_kwargs = self.generate_eval_kwargs()
        eval_generator = EvalGenerator(**eval_kwargs)
        for i in range(1, self.evaluation_chunks + 1):
            print(i)
            chunk_rewards = eval_generator.process_chunk(i)
            rewards = rewards + chunk_rewards
        return rewards

    def _parallel_evaluate(self):
        # setup process inputs
        reward_queue = mp.Queue()
        chunk_queue = mp.Queue()
        [chunk_queue.put(i) for i in range(1, self.evaluation_chunks + 1)]
        eval_kwargs = self.generate_eval_kwargs()

        # start processes
        procs = []
        for i in range(self.cpu_count):
            keywords = {
                'chunk_queue': chunk_queue,
                'reward_queue': reward_queue,
                'generator_kwargs': eval_kwargs
            }
            p = mp.Process(target=perform_eval, kwargs=keywords)
            procs.append(p)
            p.start()

        # wait until processes finish
        for p in procs:
            p.join()

        # accumulate awards
        assert reward_queue.qsize() == self.evaluation_chunks
        rewards = list()
        while not reward_queue.empty():
            rewards = rewards + reward_queue.get_nowait()
        return rewards

    def generate_eval_kwargs(self):
        args = {
            'composer': self.env_params_train['composer'],
            'model_kwargs': self.generate_model_kwargs(),
            'model_class': PgCategoricalAgentModel,
            'run_dir': self.run_dir,
            'record': False,
            'itr': self.itr,
            'verbose': self.verbose
        }
        return args

    def update_checkpoint(self):
        params = torch.load('{}params.pkl'.format(self.run_dir))
        self.checkpoint[AGENT_STATE] = params[AGENT_STATE]
        self.checkpoint[OPTIM_STATE] = params[OPTIM_STATE]

    def lr_update_needed(self, last_eval=None):
        if self.itr == 0:
            needed = False
        else:
            diff = (last_eval - self.eval_scores[-1]) / self.eval_scores[-1]
            needed = diff < FTOL
        self.eval_scores.append(last_eval)
        return needed

    def update_lr(self):
        self.lr = self.lr / 10
        assert 'lr' in self.checkpoint[OPTIM_STATE]['param_groups'][0]
        self.checkpoint[OPTIM_STATE]['param_groups'][0]['lr'] = self.lr

    def count_eval_chunks(self):
        if self.debug:
            return 10
        contents = os.listdir(RL_EVAL_DIR)
        contents = [f for f in contents if isfile(join(RL_EVAL_DIR, f))]
        pattern = re.compile(r'[0-9]+\.gz')
        contents = [f for f in contents if re.match(pattern, f)]
        return len(contents)


def perform_eval(generator_kwargs=None, chunk_queue=None, reward_queue=None):
    """
    Target function of parallel evaluation worker processes, generates
    rewards for some subset of evaluation chunks
    :param dict generator_kwargs: dictionary of kwargs for EvalGenerator
    :param mp.Queue chunk_queue: queue containing chunks that need to be processed
    :param mp.Queue reward_queue: queue that accumulates calculated rewards
    """
    generator = EvalGenerator(**generator_kwargs)
    while True:
        try:
            chunk = chunk_queue.get_nowait()
        except queue.Empty:
            break
        else:
            rewards = generator.process_chunk(chunk=chunk)
            reward_queue.put(rewards)


def main():
    parser = argparse.ArgumentParser()
    # basic arguments
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    # experiment parameters
    parser.add_argument('--batch_size', required=True, type=int)
    parser.add_argument('--batches_per_evaluation', required=True, type=int)
    parser.add_argument('--ppo_minibatches', required=True, type=int)
    parser.add_argument('--ppo_epochs', required=True, type=int)

    # Add params: entropy coefficient, value coefficient,
    # epochs per optimization step, minibatches per optimization step,
    # batch size

    # constants
    ############################################
    delay = False
    feat_id = ALL_FEATS
    con = QUARTILES
    ###########################################

    args = parser.parse_args()
    agent_params = {
        FEAT_TYPE: feat_id,
        BYR_PREFIX: False,
        CON_TYPE: con,
        DELAY: delay
    }
    trainer = RlTrainer(debug=args.debug, verbose=args.verbose,
                        agent_params=agent_params, batch_size=args.batch_size,
                        batches_per_evaluation=args.batches_per_evaluation,
                        ppo_epochs=args.ppo_epochs,
                        ppo_minibatches=args.ppo_minibatches)
    trainer.train()


if __name__ == '__main__':
    main()

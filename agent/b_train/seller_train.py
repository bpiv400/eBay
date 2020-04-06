"""
Train a seller agent that makes concessions, not offers
"""
import argparse
import os
import time
import re
import shutil
from os.path import isfile, join
import multiprocessing as mp
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
from agent.agent_consts import (BATCH_T, BATCH_B, CON_TYPE, BATCHES_PER_EVALUATION,
                                ALL_FEATS, AGENT_STATE, OPTIM_STATE,
                                TOTAL_STEPS, PPO_MINIBATCHES, SELLER_TRAIN_INPUT,
                                PPO_EPOCHS, FEAT_TYPE, BATCH_SIZE)
from agent.agent_utils import load_init_model, detect_norm
from agent.models.PgCategoricalAgentModel import PgCategoricalAgentModel
from agent.runners.EbayRunner import EbayRunner
from rlenv.env_utils import load_chunk
from rlenv.Composer import AgentComposer
from rlenv.interfaces.PlayerInterface import SimulatedBuyer, SimulatedSeller
from rlenv.interfaces.ArrivalInterface import ArrivalInterface
from rlenv.environments.SellerEnvironment import SellerEnvironment


class RlTrainer:
    def __init__(self, **kwargs):
        # arguments
        self.debug = kwargs['debug']
        self.verbose = kwargs['verbose']
        self.agent_params = kwargs['agent_params']

        # fields
        self.itr = 0
        self.norm = None
        self.evaluation_chunks = self.count_eval_chunks()
        self.checkpoint = self.init_checkpoint()

        # ids
        self.run_id = self.generate_run_id() # TODO: Make util function
        self.run_dir = '{}/run_{}/'.format(self.log_dir, self.run_id)

        # parameters
        self.env_params_train = self.generate_train_params()
        self.logger_params = self.generate_logger_params()

        # rlpyt components
        self.algorithm = self.generate_algorithm()
        self.sampler = self.generate_sampler()
        self.agent = self.generate_agent()
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
        return "runner_test"

    def clear_log(self):
        if os.path.exists(self.run_dir):
            shutil.rmtree(self.run_dir, ignore_errors=True)

    def generate_logger_params(self):
        log_params = {
            CON_TYPE: self.agent_params[CON_TYPE],
            FEAT_TYPE: self.agent_params[FEAT_TYPE],
            BYR_PREFIX: False,
            DELAY: False,
            'steps': TOTAL_STEPS,
            'ppo_minibatches': PPO_MINIBATCHES,
            'ppo_epochs': PPO_EPOCHS,
            'batch_B': BATCH_B,
            'batch_T': BATCH_T,
        }
        return log_params

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
        return PPO(minibatches=PPO_MINIBATCHES,
                   epochs=PPO_EPOCHS,
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
        if self.debug:
            return SerialSampler(
                EnvCls=SellerEnvironment,
                env_kwargs=self.env_params_train,
                batch_B=BATCH_B,
                batch_T=BATCH_T,
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
                batch_B=BATCH_B,
                batch_T=BATCH_T,
                max_decorrelation_steps=0,
                CollectorCls=CpuResetCollector,
                eval_n_envs=0,
                eval_CollectorCls=CpuEvalCollector,
                eval_env_kwargs={},
                eval_max_steps=50,
            )

    def generate_runner(self):
        # TODO: Remove constants as arguments
        runner = EbayRunner(algo=self.algorithm,
                            agent=self.agent,
                            sampler=self.sampler,
                            batches_per_evaluation=BATCHES_PER_EVALUATION,
                            batch_size=BATCH_SIZE,
                            affinity=dict(workers_cpus=list(range(4))))
        return runner

    @property
    def log_dir(self):
        extension = BYR_PREFIX if self.agent_params[BYR_PREFIX] else SLR_PREFIX
        return os.path.join(RL_LOG_DIR, extension)

    def train(self):
        with logger_context(log_dir=self.log_dir, name='debug', use_summary_writer=False,
                            override_prefix=True, run_ID=self.run_id,
                            log_params=self.logger_params, snapshot_mode='last'):
            logger.set_tf_summary_writer(self.writer)
            for i in range(10):
                self.itr = self.runner.train()
                self.writer.flush()
                self.evaluate()
                self.update_checkpoint()
                self.runner.update_agent(self.generate_agent())
                self.runner.update_algo(self.generate_algorithm())

    def evaluate(self):
        if self.debug:
            rewards, eval_time = self._serial_evaluate()
        else:
            rewards, eval_time = self._parallel_evaluate()
        self._log_eval(rewards=rewards, eval_time=time)

    def _log_eval(self, rewards=None, eval_time=None):
        logger.set_iteration(self.itr)
        logger.record_tabular('evalTime', eval_time)
        logger.record_tabular_misc_stat('evalReward', rewards)

    def _serial_evaluate(self):
        start_time = time.time()
        rewards = list()
        eval_kwargs = self.generate_eval_kwargs()
        eval_generator = EvalGenerator(**eval_kwargs)
        for i in range(10):
            eval_generator.load_chunk(chunk=i)
            if not eval_generator.initialized:
                eval_generator.initialize()
            chunk_rewards = eval_generator.generate()
            rewards = rewards + chunk_rewards
        end_time = time.time()
        return rewards, end_time - start_time

    def _parallel_evaluate(self):
        return None, None

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

    def reduce_lr(self):
        pass

    @staticmethod
    def count_eval_chunks():
        contents = os.listdir(RL_EVAL_DIR)
        contents = [f for f in contents if isfile(join(RL_EVAL_DIR, f))]
        pattern = re.compile(r'[0-9]+\.gz')
        contents = [f for f in contents if re.match(pattern, f)]
        return len(contents)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--con', required=True, type=str)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')

    # Add params: entropy coefficient, value coefficient,
    # epochs per optimization step, minibatches per optimization step,
    # batch size

    # constants
    ############################################
    delay = False
    feat_id = ALL_FEATS
    ###########################################

    args = parser.parse_args()
    agent_params = {
        FEAT_TYPE: feat_id,
        BYR_PREFIX: False,
        CON_TYPE: args.con,
        DELAY: delay
    }
    trainer = RlTrainer(debug=args.debug, verbose=args.verbose,
                        agent_params=agent_params)
    trainer.train()


if __name__ == '__main__':
    main()

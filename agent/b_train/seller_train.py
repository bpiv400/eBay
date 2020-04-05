"""
Train a seller agent that makes concessions, not offers
"""
import argparse
import os
from os.path import isfile, join
import re
import shutil
import torch
from torch.utils.tensorboard.writer import SummaryWriter

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
from constants import VALIDATION, RL_LOG_DIR, SLR_INIT, BYR_INIT, SLR_PREFIX
from agent.agent_consts import (BATCH_T, BATCH_B, CON_TYPE, BATCHES_PER_EVALUATION,
                                ALL_FEATS, AGENT_STATE, OPTIM_STATE,
                                TOTAL_STEPS, PPO_MINIBATCHES, SELLER_TRAIN_INPUT,
                                PPO_EPOCHS, FEAT_TYPE, BATCH_SIZE)
from agent.agent_utils import load_init_model, detect_norm
from agent.models.PgCategoricalAgentModel import PgCategoricalAgentModel
from agent.runners.EbayRunner import EbayRunner
from rlenv.env_utils import get_env_sim_dir, load_chunk
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
        self.evaluation_dir = get_env_sim_dir(VALIDATION)
        self.evaluation_chunks = self.count_eval_chunks()
        self.checkpoint = self.init_checkpoint()

        # ids
        self.run_id = self.generate_run_id() # TODO: Make util function
        self.exp_dir = '{}/run_{}/'.format(RL_LOG_DIR, self.run_id)

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
        self.writer = SummaryWriter(self.exp_dir)

    @staticmethod
    def init_checkpoint():
        return {
             AGENT_STATE: None,
             OPTIM_STATE: None
        }

    def generate_run_id(self):
        return "runner_test"

    def clear_log(self):
        if os.path.exists(self.exp_dir):
            shutil.rmtree(self.exp_dir, ignore_errors=True)

    def generate_logger_params(self):
        log_params = {
            CON_TYPE: self.agent_params[CON_TYPE],
            FEAT_TYPE: self.agent_params[FEAT_TYPE],
            SLR_PREFIX: True,
            DELAY: False,
            'steps': TOTAL_STEPS,
            'ppo_minibatches': PPO_MINIBATCHES,
            'ppo_epochs': PPO_EPOCHS,
            'batch_B': BATCH_B,
            'batch_T': BATCH_T,
        }
        return log_params

    def generate_train_params(self):
        x_lstg_cols = load_chunk(base_dir=self.evaluation_dir,
                                 num=1)[0].columns
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

    def generate_agent(self):
        model_kwargs = {
            'sizes': self.env_params_train['composer'].agent_sizes,
        }
        # load simulator model to initialize policy
        if self.itr == 0:
            if self.agent_params[SLR_PREFIX]:
                init_model = SLR_INIT
            else:
                init_model = BYR_INIT
            init_dict = load_init_model(name=init_model,
                                        size=model_kwargs['sizes']['out'])
            self.norm = detect_norm(init_dict)
            model_kwargs['init_dict'] = init_dict
        # set norm type
        model_kwargs['norm'] = self.norm

        return CategoricalPgAgent(ModelCls=PgCategoricalAgentModel,
                                  model_kwargs=model_kwargs,
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

    def train(self):
        with logger_context(log_dir=RL_LOG_DIR, name='debug', use_summary_writer=False,
                            override_prefix=True, run_ID=self.run_id,
                            log_params=self.logger_params, snapshot_mode='last'):
            logger.set_tf_summary_writer(self.writer)
            for i in range(10):
                self.runner.train()
                self.writer.flush()
                self.itr += BATCHES_PER_EVALUATION
                self.evaluate()
                self.update_checkpoint()
                self.runner.update_agent(self.generate_agent())
                self.runner.update_algo(self.generate_algorithm())

    def evaluate(self):
        for i in range(1, self.evaluation_chunks + 1):
            pass

    def update_checkpoint(self):
        params = torch.load('{}params.pkl'.format(self.exp_dir))
        self.checkpoint[AGENT_STATE] = params['agent_state_dict']
        self.checkpoint[OPTIM_STATE] = params['optimizer_state_dict']

    def reduce_lr(self):
        pass

    def count_eval_chunks(self):
        contents = os.listdir(self.evaluation_dir)
        contents = [f for f in contents if isfile(join(self.evaluation_dir, f))]
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
        SLR_PREFIX: True,
        CON_TYPE: args.con,
        DELAY: delay
    }
    trainer = RlTrainer(debug=args.debug, verbose=args.verbose,
                        agent_params=agent_params)
    trainer.train()


if __name__ == '__main__':
    main()

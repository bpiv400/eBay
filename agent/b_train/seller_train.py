"""
Train a seller agent that makes concessions, not offers
"""
import argparse
import os
import shutil
from torch.utils.tensorboard.writer import SummaryWriter

from rlpyt.runners.minibatch_rl import MinibatchRl
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
from constants import VALIDATION, RL_LOG_DIR
from agent.agent_consts import (BATCH_T, BATCH_B, CON_TYPE, ALL_FEATS,
                                TOTAL_STEPS, PPO_MINIBATCHES, SELLER_TRAIN_INPUT,
                                PPO_EPOCHS, FEAT_TYPE, LOG_INTERVAL_STEPS)
from agent.models.PgCategoricalAgentModel import PgCategoricalAgentModel
from constants import SLR_PREFIX
from rlenv.env_utils import get_env_sim_dir, load_chunk
from rlenv.Composer import AgentComposer
from rlenv.interfaces.PlayerInterface import BuyerInterface, SellerInterface
from rlenv.interfaces.ArrivalInterface import ArrivalInterface
from rlenv.environments.SellerEnvironment import SellerEnvironment


class RlTrainer:
    def __init__(self, **kwargs):
        self.debug = kwargs['debug']
        self.verbose = kwargs['verbose']
        self.agent_params = kwargs['agent_params']
        self.run_id = self.generate_run_id() # TODO: Make util function
        self.exp_dir = '{}/run_{}/'.format(RL_LOG_DIR, self.run_id)

        # environment parameters
        self.env_params_train = self.generate_train_params()
        self.logger_params = self.generate_logger_params()

        # rlpyt components
        self.algorithm = self.generate_algorithm()
        self.sampler = self.generate_sampler()
        self.agent = self.generate_agent()
        self.runner = self.generate_runner()

        self.writer = SummaryWriter(self.exp_dir)
        self.clear_log()

    def generate_run_id(self):
        return "default"

    def clear_log(self):
        if os.path.exists(self.exp_dir):
            shutil.rmtree(self.exp_dir)

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
        x_lstg_cols = load_chunk(base_dir=get_env_sim_dir(VALIDATION),
                                 num=1)[0].columns
        composer = AgentComposer(cols=x_lstg_cols,
                                 agent_params=self.agent_params)
        env_params = {
            'composer': composer,
            'verbose': self.verbose,
            'filename': SELLER_TRAIN_INPUT,
            'arrival': ArrivalInterface(),
            'seller': SellerInterface(full=False),
            'buyer': BuyerInterface()
        }
        return env_params

    @staticmethod
    def generate_algorithm():
        return PPO(minibatches=PPO_MINIBATCHES,
                   epochs=PPO_EPOCHS)

    def generate_agent(self):
        model_kwargs = {
            'sizes': self.env_params_train['composer'].agent_sizes,
        }
        return CategoricalPgAgent(ModelCls=PgCategoricalAgentModel,
                                  model_kwargs=model_kwargs)

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
        runner = MinibatchRl(log_traj_window=100,
                             algo=self.algorithm,
                             agent=self.agent,
                             sampler=self.sampler,
                             n_steps=TOTAL_STEPS,
                             log_interval_steps=LOG_INTERVAL_STEPS,
                             affinity=dict(workers_cpus=list(range(4))))
        return runner

    def train(self):
        with logger_context(log_dir=RL_LOG_DIR, name='debug', use_summary_writer=True,
                            override_prefix=True, run_ID=self.run_id,
                            log_params=self.logger_params, snapshot_mode='last'):
            logger.set_tf_summary_writer(self.writer)
            for i in range(10):
                self.runner.train()

    def validate(self):
        pass

    def reduce_lr(self):
        pass


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

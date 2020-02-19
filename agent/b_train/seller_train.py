import argparse
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.algos.pg.ppo import PPO
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector
from rlpyt.samplers.serial.sampler import SerialEvalCollector
from featnames import DELAY
from constants import VALIDATION
from agent.agent_consts import (BATCH_T, BATCH_B, CON_TYPE,
                                TOTAL_STEPS, PPO_MINIBATCHES,
                                PPO_EPOCHS, FEAT_ID)
from agent.agent_utils import slr_input_path
from agent.models.PgCategoricalAgentModel import PgCategoricalAgentModel
from rlenv.env_consts import SLR_PREFIX
from rlenv.env_utils import get_env_sim_dir, load_chunk
from rlenv.Composer import AgentComposer
from rlenv.interfaces.PlayerInterface import BuyerInterface, SellerInterface
from rlenv.interfaces.ArrivalInterface import ArrivalInterface
from rlenv.environments.SellerEnvironment import SellerEnvironment


def make_agent(env_params=None):
    model_kwargs = {
        'sizes': env_params['composer'].agent_sizes,
        'delay': env_params['composer'].delay,
    }
    return CategoricalPgAgent(ModelCls=PgCategoricalAgentModel,
                              model_kwargs=model_kwargs)


def make_algo(env_params=None):
    return PPO(minibatches=PPO_MINIBATCHES,
               epochs=PPO_EPOCHS)


def make_sampler(env_params=None):
    return SerialSampler(
        EnvCls=SellerEnvironment,
        env_kwargs=env_params,
        batch_B=BATCH_B,
        batch_T=BATCH_T,
        max_decorrelation_steps=0,
        TrajInfoCls=None,
        CollectorCls=CpuResetCollector,
        eval_n_envs=1,
        eval_CollectorCls=SerialEvalCollector,
        eval_env_kwargs=env_params,
        eval_max_steps=50,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', required=True, type=str)
    parser.add_argument('--con', required=True, type=str)
    parser.add_argument('--verbose', action='store_true')

    # TODO: Develop multi-dimensional delay model
    # parser.add_argument('--delay', action='store_true')
    delay = False
    ###########################################

    args = parser.parse_args()

    agent_params = {
        FEAT_ID: 0,
        SLR_PREFIX: True,
        CON_TYPE: args.con,
        DELAY: delay
    }

    x_lstg_cols = load_chunk(base_dir=get_env_sim_dir(VALIDATION),
                             num=1)[0].columns
    composer = AgentComposer(cols=x_lstg_cols, agent_params=agent_params)

    env_params = {
        'composer': composer,
        'verbose': args.verbose,
        'filename': slr_input_path(args.part),
        'arrival': ArrivalInterface(),
        'seller': SellerInterface(full=False),
        'buyer': BuyerInterface()
    }

    agent = make_agent(env_params=env_params)
    algo = make_algo(env_params=env_params)
    sampler = make_sampler(env_params)

    runner = MinibatchRl(log_traj_window=100,
                         algo=algo,
                         agent=agent,
                         sampler=sampler,
                         n_steps=TOTAL_STEPS)
    # not sure if this is right
    runner.train()

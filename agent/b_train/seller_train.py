import argparse
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.algos.pg.ppo import PPO
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from agent.agent_consts import (BATCH_T, BATCH_B, CON_TYPE,
                                TOTAL_STEPS, PPO_MINIBATCHES,
                                PPO_EPOCHS)


def make_agent(params):
    CategoricalPgAgent(ModelCls=None,
                       )


def make_algo(params):
    return PPO(minibatches=PPO_MINIBATCHES,
               epochs=PPO_EPOCHS)


def make_sampler(params):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', required=True, type=str)
    parser.add_argument('--con', required=True, type=str)
    parser.add_argument('--delay', action='store_true')
    args = parser.parse_args()

    con_set = get_con_set(args.con)

    params = {
        'part': args.part,
        FEAT_ID: 0,
        SLR_PREFIX: True,
        CON_TYPE: args.con,
        'delay': args.delay
    }
    agent = make_agent(params)
    algo = make_algo(params)
    sampler = make_sampler(params)

    runner = MinibatchRl(log_traj_window=100,
                         algo=algo,
                         agent=agent,
                         sampler=sampler,
                         n_steps=TOTAL_STEPS)

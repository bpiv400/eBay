import argparse
import os
import pandas as pd
from datetime import datetime as dt
from compress_pickle import load, dump
from agent.eval.EvalGenerator import EvalGenerator
from agent.models.PgCategoricalAgentModel import PgCategoricalAgentModel
from agent.AgentComposer import AgentComposer
from rlenv.env_utils import load_chunk
from agent.agent_consts import AGENT_PARAMS, NO_ARRIVAL, NO_ARRIVAL_CUTOFF
from constants import RL_EVAL_DIR, BYR_PREFIX, SLR_PREFIX, \
    RL_LOG_DIR, RL_NORM
from featnames import DELAY


def gen_eval_kwargs(composer=None, model_kwargs=None,
                    run_dir=None, itr=None, num=None, record=None):
    eval_kwargs = {'composer': composer,
                   'model_kwargs': model_kwargs,
                   'model_class': PgCategoricalAgentModel,
                   'run_dir': run_dir,
                   'itr': itr,
                   'num': num,
                   'record': record,
                   'verbose': False}
    return eval_kwargs


def gen_model_kwargs(composer):
    model_kwargs = {'sizes': composer.agent_sizes,
                    BYR_PREFIX: composer.byr,
                    DELAY: composer.delay,
                    'norm': RL_NORM}
    return model_kwargs


def main():
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--byr', action='store_true')
    parser.add_argument('--num', type=int, required=True)
    parser.add_argument('--run', type=str, required=True)
    args = parser.parse_args()
    num, run = args.num, args.run
    role = BYR_PREFIX if args.byr else SLR_PREFIX

    # load chunk
    chunk_path = '{}{}.gz'.format(RL_EVAL_DIR, num)
    chunk = load_chunk(input_path=chunk_path)

    # restrict to no arrivals
    keep = chunk[1][NO_ARRIVAL] < NO_ARRIVAL_CUTOFF
    idx = chunk[1].index[keep]
    chunk = [df.reindex(index=idx) for df in chunk]

    # find run parameters
    parent_dir = RL_LOG_DIR + '{}/'.format(role)
    params = load(parent_dir + 'runs.pkl').loc[run]
    run_dir = parent_dir + 'run_{}/'.format(run)

    # create composer
    agent_params = params[AGENT_PARAMS.keys()]
    composer = AgentComposer(cols=chunk[0].columns,
                             agent_params=agent_params)
    model_kwargs = gen_model_kwargs(composer)

    # create outcomes directory
    outcomes_dir = run_dir + 'outcomes/'
    if not os.path.isdir(outcomes_dir):
        os.mkdir(outcomes_dir)

    # loop over models
    iterations = int(params['batch_count'])
    for itr in range(iterations):
        # rewards directory
        rewards_dir = run_dir + 'itr/{}/rewards/'.format(itr)
        if not os.path.isdir(rewards_dir):
            os.makedirs(rewards_dir)

        # check if file exists
        if os.path.isfile(rewards_dir + '{}.gz'.format(num)):
            print('Rewards for iteration {} already exists.'.format(itr))
            continue

        # timer
        t0 = dt.now()

        # create generator
        eval_kwargs = gen_eval_kwargs(composer=composer,
                                      model_kwargs=model_kwargs,
                                      run_dir=run_dir,
                                      itr=itr,
                                      num=num,
                                      record=(itr == iterations - 1))
        eval_generator = EvalGenerator(**eval_kwargs)

        # run generator to simulate rewards
        rewards = eval_generator.process_chunk(chunk)
        rewards = pd.Series(rewards, index=chunk[0].index)

        # print elapsed time
        seconds = (dt.now() - t0).total_seconds()
        print('Iteration {0:d}: {1:.0f} seconds'.format(itr, seconds))

        # save rewards
        dump(rewards, rewards_dir + '{}.gz'.format(num))


if __name__ == '__main__':
    main()

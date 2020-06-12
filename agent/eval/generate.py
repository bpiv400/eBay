import argparse
import os
from compress_pickle import load
from agent.eval.EvalGenerator import EvalGenerator
from agent.models.PgCategoricalAgentModel import PgCategoricalAgentModel
from agent.AgentComposer import AgentComposer
from rlenv.utils import load_chunk
from agent.agent_consts import AGENT_PARAMS, DROPOUT
from constants import NO_ARRIVAL_CUTOFF, BYR_PREFIX, SLR_PREFIX, \
    RL_LOG_DIR, TRAIN_RL, VALIDATION, TEST, PARTS_DIR
from featnames import DELAY, NO_ARRIVAL


def gen_eval_kwargs(composer=None, model_kwargs=None, run_dir=None,
                    path_suffix=None):
    eval_kwargs = {'composer': composer,
                   'model_kwargs': model_kwargs,
                   'model_class': PgCategoricalAgentModel,
                   'run_dir': run_dir,
                   'path_suffix': path_suffix,
                   'verbose': False}
    return eval_kwargs


def gen_model_kwargs(agent_params):
    model_kwargs = {BYR_PREFIX: agent_params.role == BYR_PREFIX,
                    DELAY: agent_params.delay,
                    DROPOUT: (agent_params.dropout0,
                              agent_params.dropout1)}
    return model_kwargs


def main():
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--byr', action='store_true')
    parser.add_argument('--num', type=int, required=True)
    parser.add_argument('--part', type=str, required=True,
                        choices=[TRAIN_RL, VALIDATION, TEST])
    args = parser.parse_args()
    role = BYR_PREFIX if args.byr else SLR_PREFIX

    # load chunk
    chunk_path = PARTS_DIR + '{}/chunks/{}.gz'.format(args.part, args.num)
    chunk = load_chunk(input_path=chunk_path)

    # restrict to no arrivals
    keep = chunk[1][NO_ARRIVAL] < NO_ARRIVAL_CUTOFF
    idx = chunk[1].index[keep]
    chunk = [df.reindex(index=idx) for df in chunk]

    # find run parameters
    parent_dir = RL_LOG_DIR + '{}/'.format(role)
    params = load(parent_dir + 'runs.pkl')
    runs = params.index

    # loop over runs
    for run in runs:
        # run directory
        run_dir = parent_dir + 'run_{}/'.format(run)

        # check if file exists
        path_suffix = '{}/{}.gz'.format(args.part, args.num)
        if os.path.isfile(run_dir + path_suffix):
            print('Outcomes for [run {} / chunk {}] already generated.'.format(
                run, args.num))
            continue
        else:
            if not os.path.isdir(run_dir + args.part):
                os.mkdir(run_dir + args.part)
            print(run_dir + path_suffix)

        # create composer
        agent_params = params.loc[run, AGENT_PARAMS.keys()]
        composer = AgentComposer(cols=chunk[0].columns,
                                 agent_params=agent_params)
        model_kwargs = gen_model_kwargs(agent_params)

        # create generator
        eval_kwargs = gen_eval_kwargs(composer=composer,
                                      model_kwargs=model_kwargs,
                                      run_dir=run_dir,
                                      path_suffix=path_suffix)
        eval_generator = EvalGenerator(**eval_kwargs)

        # run generator to simulate outcomes
        eval_generator.process_chunk(chunk)


if __name__ == '__main__':
    main()

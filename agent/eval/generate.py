import argparse
import os
from compress_pickle import load
from agent.eval.EvalGenerator import EvalGenerator
from agent.AgentComposer import AgentComposer
from agent.const import ALL_FEATS, FULL_CON
from constants import MODEL_DIR, BYR, RL_LOG_DIR, TRAIN_RL, VALIDATION, \
    TEST, AGENTS
from featnames import DELAY


def gen_eval_kwargs(part=None, composer=None, model_kwargs=None, run_dir=None):
    eval_kwargs = {'part': part,
                   'composer': composer,
                   'model_kwargs': model_kwargs,
                   'run_dir': run_dir,
                   'verbose': False}
    return eval_kwargs


def gen_model_kwargs(name=None):
    model_kwargs = {BYR: BYR in name,
                    DELAY: DELAY in name}
    # add in dropout
    s = load(MODEL_DIR + 'dropout.pkl')
    for net in ['policy', 'value']:
        model_kwargs['dropout_{}'.format(net)] = \
            s.loc['{}_{}'.format(net, name)]
    return model_kwargs


def gen_agent_params(name):
    agent_params = {'name': name,
                    'feat_id': ALL_FEATS,
                    'con_type': FULL_CON}
    return agent_params


def main():
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', choices=AGENTS, required=True)
    parser.add_argument('--num', type=int, required=True)
    parser.add_argument('--part', type=str, required=True,
                        choices=[TRAIN_RL, VALIDATION, TEST])
    args = parser.parse_args()
    name, num, part = args.name, args.num, args.part

    # find run parameters
    parent_dir = RL_LOG_DIR + '{}/'.format(name)
    params = load(parent_dir + 'runs.pkl')
    runs = params.index

    # loop over runs
    for run in runs:
        # run directory
        run_dir = parent_dir + 'run_{}/'.format(run)

        # check if file exists
        path_suffix = '{}/{}.gz'.format(part, num)
        if os.path.isfile(run_dir + path_suffix):
            print('run_{}/{}.gz already simulated.'.format(run, num))
            continue
        else:
            if not os.path.isdir(run_dir + part):
                os.mkdir(run_dir + part)
            print(run_dir + path_suffix)

        # create composer
        agent_params = gen_agent_params(name)
        composer = AgentComposer(agent_params=agent_params)

        # create generator
        model_kwargs = gen_model_kwargs(name)
        eval_kwargs = gen_eval_kwargs(part=part,
                                      composer=composer,
                                      model_kwargs=model_kwargs,
                                      run_dir=run_dir)
        eval_generator = EvalGenerator(**eval_kwargs)

        # run generator to simulate outcomes
        eval_generator.process_chunk(num)


if __name__ == '__main__':
    main()

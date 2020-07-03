import os
import argparse
import pandas as pd
from datetime import datetime as dt
import torch
from compress_pickle import load, dump
from agent.models.PgCategoricalAgentModel import PgCategoricalAgentModel
from agent.RlTrainer import RlTrainer
from agent.const import AGENT_STATE, PARAM_DICTS
from agent.util import compose_args
from agent.eval.EvalGenerator import EvalGenerator
from utils import set_gpu_workers
from constants import MODEL_DIR, RL_LOG_DIR, TRAIN_RL, VALIDATION

CHUNK_NUM = 1


def simulate(part=None, run_dir=None, composer=None, model_kwargs=None):
    eval_kwargs = {'part': part,
                   'composer': composer,
                   'model_kwargs': model_kwargs,
                   'model_class': PgCategoricalAgentModel,
                   'run_dir': run_dir,
                   'verbose': False}
    eval_generator = EvalGenerator(**eval_kwargs)
    # TODO: run in parallel over all chunks
    eval_generator.process_chunk(chunk=CHUNK_NUM, drop_infreq=True)


def main():
    set_gpu_workers()  # set gpu and cpu affinity

    # command-line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int)
    for d in PARAM_DICTS.values():
        compose_args(arg_dict=d, parser=parser)
    args = vars(parser.parse_args())

    # swap in experiment parameters
    if args['exp'] is not None:
        print('Using parameters for experiment {}'.format(args['exp']))
        exp_path = RL_LOG_DIR + 'exps.pkl'
        params = load(exp_path).loc[args['exp']].to_dict()
        for k, v in params.items():
            if k in args:
                args[k] = v
    else:
        params = None
    del args['exp']

    # add dropout
    s = load(MODEL_DIR + 'dropout.pkl')
    for name in ['policy', 'value']:
        args['dropout_{}'.format(name)] = \
            s.loc['{}_{}'.format(name, args['name'])]

    # print to console
    for k, v in args.items():
        print('{}: {}'.format(k, v))

    # split parameters
    trainer_args = dict()
    for param_set, param_dict in PARAM_DICTS.items():
        curr_params = dict()
        for k in param_dict.keys():
            curr_params[k] = args[k]
        trainer_args[param_set] = curr_params

    # initialize trainer
    trainer = RlTrainer(**trainer_args)

    # training loop
    t0 = dt.now()
    trainer.train()
    time_elapsed = (dt.now() - t0).total_seconds()

    # run directory
    run_dir = trainer.log_dir + 'run_{}/'.format(trainer.run_id)

    # drop optimization parameters
    if not args['no_logging']:
        path = run_dir + 'params.pkl'
        d = torch.load(path)
        torch.save(d[AGENT_STATE], path)

        # simulate outcomes
        for part in [TRAIN_RL, VALIDATION]:
            os.mkdir(run_dir + '{}/'.format(part))
            simulate(part=part,
                     run_dir=run_dir,
                     composer=trainer.composer,
                     model_kwargs=trainer.model_params)

    # save experiment results
    if params is not None:
        # load (or create) runs file
        run_path = trainer.log_dir + 'runs.pkl'
        if os.path.isfile(run_path):
            df = load(run_path)
        else:
            df = pd.DataFrame(index=pd.Index([], name='run_id'))
        for k, v in params.items():
            df.loc[trainer.run_id, k] = v
        df.loc[trainer.run_id, 'seconds'] = int(time_elapsed)
        dump(df, run_path)


if __name__ == '__main__':
    main()

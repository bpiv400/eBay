import argparse
import os
import torch
import pandas as pd
from agent.RlTrainer import RlTrainer
from agent.const import PARAM_DICTS
from agent.eval.EvalGenerator import EvalGenerator
from agent.util import get_log_dir, get_run_id, get_values
from rlenv.generate.util import process_sims
from utils import unpickle, topickle, compose_args, set_gpu_workers, \
    run_func_on_chunks, process_chunk_worker
from constants import DROPOUT_PATH, POLICY_BYR, BYR, POLICY_SLR, DROPOUT, \
    VALIDATION, NUM_RL_WORKERS


def save_values(log_dir=None, run_id=None, values=None):
    path = log_dir + 'runs.pkl'
    if os.path.isfile(path):
        df = unpickle(path)
    else:
        df = pd.DataFrame(index=pd.Index([], name='run_id'))

    # moments of value distribution
    for col in values.columns:
        s = values[col]
        df.loc[run_id, '{}_mean'.format(col)] = s.mean()
        df.loc[run_id, '{}_median'.format(col)] = s.median()
        df.loc[run_id, '{}_min'.format(col)] = s.min()
        df.loc[run_id, '{}_max'.format(col)] = s.max()
        df.loc[run_id, '{}_std'.format(col)] = s.std()

    # save
    topickle(contents=df, path=path)


def simulate(part=None, trainer=None):
    # directory for simulation output
    run_dir = trainer.log_dir + '{}/'.format(trainer.run_id)
    part_dir = run_dir + '{}/'.format(part)
    if not os.path.isdir(part_dir):
        os.mkdir(part_dir)

    # arguments for generator
    eval_kwargs = dict(
        byr=trainer.byr,
        dropout=trainer.model_params[DROPOUT],
        run_dir=run_dir
    )

    # run in parallel on chunks
    sims = run_func_on_chunks(
        num_chunks=NUM_RL_WORKERS,
        f=process_chunk_worker,
        func_kwargs=dict(
            part=part,
            gen_class=EvalGenerator,
            gen_kwargs=eval_kwargs
        )
    )

    # combine and process output
    process_sims(part=part, parent_dir=run_dir, sims=sims)

    # evaluate simulations
    values = get_values(part=VALIDATION,
                        run_dir=run_dir,
                        prefs=trainer.algo.prefs)
    save_values(values)


def get_model_params(**args):
    # initialize with role an dropout
    s = unpickle(DROPOUT_PATH)
    dropout = s.loc[POLICY_BYR if args[BYR] else POLICY_SLR]
    model_params = {BYR: args[BYR], DROPOUT: dropout}
    print('{}: {}'.format(DROPOUT, dropout))

    # if using cross entropy, load entropy model
    if args['kl_coeff'] is not None:
        entropy_dir = get_log_dir(**args)
        entropy_id = get_run_id(**args)
        model_path = '{}_{}/params.pkl'.format(entropy_dir, entropy_id)
        state_dict = torch.load(model_path,
                                map_location=torch.device('cpu'))
        model_params['model_state_dict'] = state_dict

    return model_params


def startup():
    # command-line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--byr', action='store_true')
    for d in PARAM_DICTS.values():
        compose_args(arg_dict=d, parser=parser)
    args = vars(parser.parse_args())

    # set gpu and cpu affinity
    set_gpu_workers(gpu=args['gpu'], spawn=True)

    # print to console
    for k, v in args.items():
        print('{}: {}'.format(k, v))

    # split parameters
    trainer_args = dict()
    for param_set, param_dict in PARAM_DICTS.items():
        trainer_args[param_set] = {k: args[k] for k in param_dict}

    # model parameters
    trainer_args['model'] = get_model_params(**args)

    return trainer_args


def main():
    trainer_args = startup()
    trainer = RlTrainer(**trainer_args)
    trainer.train()
    simulate(part=VALIDATION, trainer=trainer)


if __name__ == '__main__':
    main()

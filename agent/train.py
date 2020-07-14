from multiprocessing import set_start_method
import os
import argparse
import pandas as pd
import torch
from compress_pickle import load
from agent.RlTrainer import RlTrainer
from agent.const import AGENT_STATE, PARAM_DICTS, AGENT_PARAMS, SYSTEM_PARAMS
from agent.util import compose_args
from agent.eval.EvalGenerator import EvalGenerator
from utils import set_gpu_workers
from constants import MODEL_DIR, AGENT_DIR, TRAIN_RL, VALIDATION, \
    POLICY_SLR, POLICY_BYR, BYR, DROPOUT


def simulate(part=None, run_dir=None, composer=None, model_kwargs=None):
    eval_kwargs = {'part': part,
                   'composer': composer,
                   'model_kwargs': model_kwargs,
                   'run_dir': run_dir,
                   'verbose': False}
    eval_generator = EvalGenerator(**eval_kwargs)
    # TODO: run in parallel over all chunks
    eval_generator.process_chunk(chunk=1)


def main():
    # command-line parameters
    parser = argparse.ArgumentParser()
    for d in PARAM_DICTS.values():
        compose_args(arg_dict=d, parser=parser)
    args = vars(parser.parse_args())

    # set gpu and cpu affinity
    set_gpu_workers(gpu=args['gpu'])

    # swap in experiment parameters
    if args['exp'] is not None:
        print('Using parameters for experiment {}'.format(args['exp']))
        exp_path = AGENT_DIR + 'exps.csv'
        params = pd.read_csv(exp_path, index_col=0).loc[args['exp']].to_dict()
        for k, v in params.items():
            if k not in args:
                raise RuntimeError('{} not in args'.format(k))
            args[k] = v

    # add dropout
    # s = load(MODEL_DIR + 'dropout.pkl')
    # args[DROPOUT] = s.loc[POLICY_BYR if args['byr'] else POLICY_SLR]
    args[DROPOUT] = (0., 0.)

    # print to console
    for k, v in args.items():
        print('{}: {}'.format(k, v))

    # split parameters
    trainer_args = dict()
    for param_set, param_dict in PARAM_DICTS.items():
        curr_params = dict()
        for k in param_dict:
            if k in args:
                curr_params[k] = args[k]
        trainer_args[param_set] = curr_params

    # model parameters
    trainer_args['model_params'] = {BYR: args[BYR],
                                    DROPOUT: args[DROPOUT]}

    # training loop
    trainer = RlTrainer(**trainer_args)
    trainer.train()

    # when logging, simulate and reconfigure outputs
    if args['log']:
        run_dir = trainer.log_dir + '/run_{}/'.format(trainer.run_id)

        # drop optimization parameters
        path = run_dir + 'params.pkl'
        d = torch.load(path)
        torch.save(d[AGENT_STATE], path)

        # save run parameters
        run_path = trainer.log_dir + 'runs.csv'
        if os.path.isfile(run_path):
            df = pd.read_csv(run_path, index_col=0)
        else:
            df = pd.DataFrame(index=pd.Index([], name='run_id'))

        exclude = list({**AGENT_PARAMS, **SYSTEM_PARAMS}.keys())
        exclude += ['dropout']
        exclude.remove('batch_size')
        for k, v in args.items():
            if k not in exclude:
                df.loc[trainer.run_id, k] = v
        df.to_csv(run_path)

        # # simulate outcomes
        # for part in [TRAIN_RL, VALIDATION]:
        #     os.mkdir(run_dir + '{}/'.format(part))
        #     simulate(part=part,
        #              run_dir=run_dir,
        #              composer=trainer.composer,
        #              model_kwargs=trainer.model_params)


if __name__ == '__main__':
    set_start_method("spawn")
    main()

import argparse
import os
from agent.RlTrainer import RlTrainer
from agent.const import PARAM_DICTS
from utils import compose_args, set_gpu_workers
from constants import BYR


def startup():
    # command-line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--byr', action='store_true')
    for d in PARAM_DICTS.values():
        compose_args(arg_dict=d, parser=parser)
    args = vars(parser.parse_args())

    # set gpu and cpu affinity
    set_gpu_workers(gpu=args['gpu'], use_all=args['all'], spawn=True)

    # print to console
    for k, v in args.items():
        print('{}: {}'.format(k, v))

    # split parameters
    trainer_args = dict()
    for param_set, param_dict in PARAM_DICTS.items():
        trainer_args[param_set] = {k: args[k] for k in param_dict}
    trainer_args[BYR] = args[BYR]

    return trainer_args


def main():
    trainer_args = startup()
    trainer = RlTrainer(**trainer_args)

    # if logging and model has already been trained, quit
    if trainer_args['system']['log'] and os.path.isdir(trainer.run_dir):
        print('{} already exists.'.format(trainer.run_id))
        exit()

    # train
    trainer.train()


if __name__ == '__main__':
    main()

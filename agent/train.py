import argparse
import os
from agent.RlTrainer import RlTrainer
from agent.const import PARAMS
from utils import compose_args, set_gpu


def startup():
    # command-line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--log', action='store_true')
    compose_args(arg_dict=PARAMS, parser=parser)
    args = vars(parser.parse_args())

    # set gpu and cpu affinity
    set_gpu(gpu=args['gpu'])

    # print to console
    for k, v in args.items():
        print('{}: {}'.format(k, v))

    # params for trainer initialization
    params = {k: v for k, v in args.items() if k in PARAMS.keys()}

    return params, args['log']


def main():
    params, log = startup()
    trainer = RlTrainer(**params)

    # if logging and model has already been trained, quit
    if log and os.path.isdir(trainer.run_dir):
        print('{} already exists.'.format(trainer.run_id))
        exit()

    # train
    trainer.train(log=log)


if __name__ == '__main__':
    main()

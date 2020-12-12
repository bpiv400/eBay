import argparse
import os
import torch
from agent.RlTrainer import RlTrainer
from agent.util import get_run_dir
from utils import compose_args, set_gpu
from agent.const import AGENT_PARAMS, DELTA_SLR, DELTA_BYR
from featnames import BYR, DELTA, TURN_COST


def startup():
    # command-line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--serial', action='store_true')
    compose_args(arg_dict=AGENT_PARAMS, parser=parser)
    args = vars(parser.parse_args())

    # error checking
    if args[BYR]:
        assert args[DELTA] in DELTA_BYR
    else:
        assert args[DELTA] in DELTA_SLR
        assert args[TURN_COST] == 0

    if args['serial']:
        assert not args['log']

    # make sure run doesn't already exist
    if args['log']:
        run_dir = get_run_dir(**args)
        if os.path.isdir(run_dir):
            print('Run already exists.')
            exit()

    # set gpu and cpu affinity
    set_gpu(gpu=args['gpu'])
    del args['gpu']

    # print to console
    for k, v in args.items():
        print('{}: {}'.format(k, v))

    # params for trainer initialization
    agent_params = {k: v for k, v in args.items()
                    if k in AGENT_PARAMS.keys()}
    train_params = {k: v for k, v in args.items()
                    if k not in AGENT_PARAMS.keys()}

    return agent_params, train_params


def main():
    agent_params, train_params = startup()
    trainer = RlTrainer(**agent_params)
    trainer.train(**train_params)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork')
    main()

import argparse
import torch
from agent.RlTrainer import RlTrainer
from utils import compose_args, set_gpu
from agent.const import AGENT_PARAMS, HYPER_PARAMS
from constants import DROPOUT_GRID
from featnames import DROPOUT


def startup():
    # command-line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--serial', action='store_true')
    parser.add_argument('--suffix', type=str)
    compose_args(arg_dict=AGENT_PARAMS, parser=parser)
    compose_args(arg_dict=HYPER_PARAMS, parser=parser)
    args = vars(parser.parse_args())

    # set gpu and cpu affinity
    set_gpu(gpu=args['gpu'])
    del args['gpu']

    # translate dropout index
    args[DROPOUT] = DROPOUT_GRID[args[DROPOUT]]

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

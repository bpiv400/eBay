import argparse
from agent.RlTrainer import RlTrainer
from agent.const import PARAMS
from utils import compose_args, set_gpu
from constants import DROPOUT_GRID
from featnames import DROPOUT


def startup():
    # command-line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--serial', action='store_true')
    compose_args(arg_dict=PARAMS, parser=parser)
    args = vars(parser.parse_args())

    # set gpu and cpu affinity
    set_gpu(gpu=args['gpu'])
    del args['gpu']

    # translate dropout from index to tuple
    args[DROPOUT] = DROPOUT_GRID[args[DROPOUT]]

    # print to console
    for k, v in args.items():
        print('{}: {}'.format(k, v))

    # params for trainer initialization
    init_params = {k: v for k, v in args.items() if k in PARAMS.keys()}
    train_params = {k: v for k, v in args.items() if k not in PARAMS.keys()}

    return init_params, train_params


def main():
    init_params, train_params = startup()
    trainer = RlTrainer(**init_params)

    # train
    trainer.train(**train_params)


if __name__ == '__main__':
    main()

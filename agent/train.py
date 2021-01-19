import argparse
import os
import torch
from agent.RlTrainer import RlTrainer
from agent.util import get_run_dir
from utils import set_gpu
from agent.const import DELTA_CHOICES


def main():
    # command-line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--serial', action='store_true')
    parser.add_argument('--delta', type=float, choices=DELTA_CHOICES)
    args = parser.parse_args()

    # error checking
    if args.serial:
        assert not args.log

    # make sure run doesn't already exist
    if args.log:
        run_dir = get_run_dir(delta=args.delta)
        if os.path.isdir(run_dir):
            print('Run already exists.')
            exit()

    # set gpu
    set_gpu(gpu=args.gpu)

    # print to console
    for k, v in vars(args).items():
        print('{}: {}'.format(k, v))

    trainer = RlTrainer(delta=args.delta)
    trainer.train(log=args.log, serial=args.serial)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork')
    main()

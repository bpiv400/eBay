import argparse
from agent.const import AGENT_PARAMS, HYPER_PARAMS
from agent.util import get_log_dir, get_paths
from constants import DROPOUT_GRID, NUM_CHUNKS
from featnames import DROPOUT, AGENT_PARTITIONS, TEST
from utils import compose_args


def sim_run_dir(args=None):
    if args['heuristic']:
        run_dir = get_log_dir(**args) + 'heuristic/'
    else:
        args[DROPOUT] = DROPOUT_GRID[args[DROPOUT]]
        _, _, run_dir = get_paths(**args)
    return run_dir


def sim_args(num=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str,
                        choices=AGENT_PARTITIONS, default=TEST)
    parser.add_argument('--heuristic', action='store_true')
    parser.add_argument('--suffix', type=str)
    compose_args(arg_dict=AGENT_PARAMS, parser=parser)
    compose_args(arg_dict=HYPER_PARAMS, parser=parser)
    if num:
        parser.add_argument('--num', type=int,
                            choices=range(1, NUM_CHUNKS + 1))
    args = vars(parser.parse_args())
    return args

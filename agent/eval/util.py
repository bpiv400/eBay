import argparse
from agent.const import AGENT_PARAMS
from constants import NUM_CHUNKS
from featnames import AGENT_PARTITIONS, TEST
from utils import compose_args


def sim_args(num=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str,
                        choices=AGENT_PARTITIONS, default=TEST)
    parser.add_argument('--heuristic', action='store_true')
    compose_args(arg_dict=AGENT_PARAMS, parser=parser)
    if num:
        parser.add_argument('--num', type=int,
                            choices=range(1, NUM_CHUNKS + 1))
    args = vars(parser.parse_args())
    return args

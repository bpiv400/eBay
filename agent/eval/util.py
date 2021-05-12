import argparse
from agent.util import get_log_dir
from agent.const import DELTA_CHOICES, TURN_COST_CHOICES
from constants import NUM_CHUNKS
from featnames import AGENT_PARTITIONS, TEST


def sim_args(num=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str,
                        choices=AGENT_PARTITIONS, default=TEST)
    parser.add_argument('--heuristic', action='store_true')
    parser.add_argument('--byr', action='store_true')
    parser.add_argument('--delta', type=float, choices=DELTA_CHOICES)
    parser.add_argument('--turn_cost', type=int, default=0,
                        choices=TURN_COST_CHOICES)
    if num:
        parser.add_argument('--num', type=int,
                            choices=range(1, NUM_CHUNKS + 1))
    return parser.parse_args()


def eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read', action='store_true')
    return parser.parse_args()


def get_eval_path(part=TEST, byr=None):
    return '{}{}.pkl'.format(get_log_dir(byr=byr), part)

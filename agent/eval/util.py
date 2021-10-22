import argparse
from agent.util import get_log_dir
from agent.const import DELTA_BYR, DELTA_SLR, TURN_COST_CHOICES, BYR_CONS
from constants import NUM_CHUNKS
from featnames import AGENT_PARTITIONS, TEST


def sim_args(num=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str,
                        choices=AGENT_PARTITIONS, default=TEST)
    parser.add_argument('--heuristic', action='store_true')
    parser.add_argument('--byr', action='store_true')
    parser.add_argument('--delta', type=float)
    parser.add_argument('--turn_cost', type=int, default=0,
                        choices=TURN_COST_CHOICES)
    parser.add_argument('--index', type=int, choices=BYR_CONS.index)
    if num:
        parser.add_argument('--num', type=int,
                            choices=range(1, NUM_CHUNKS + 1))
    args = parser.parse_args()
    if args.byr:
        if args.heuristic:
            assert args.delta is None
        else:
            assert args.delta in DELTA_BYR
    else:
        assert args.delta in DELTA_SLR
    return args


def eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read', action='store_true')
    return parser.parse_args()


def get_eval_path(part=TEST, byr=None):
    return '{}{}.pkl'.format(get_log_dir(byr=byr), part)

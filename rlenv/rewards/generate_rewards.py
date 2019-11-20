"""
Generates rewards for a chunk of the lstgs in a partition
"""
import argparse
import torch
from constants import PARTITIONS, REWARDS_DIR
from rlenv.rewards.RewardGenerator import RewardGenerator


def main():
    torch.set_default_dtype(torch.float32)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', required=True, help='chunk number')
    parser.add_argument('--part', required=True, help='partition name')
    parser.add_argument('--id', required=True, help='experiment id')
    args = parser.parse_args()
    part = args.part
    num = args.num
    exp = args.id
    if part not in PARTITIONS:
        raise RuntimeError('part must be one of: {}'.format(PARTITIONS))
    base_dir = '{}{}/'.format(REWARDS_DIR, part)
    generator = RewardGenerator(base_dir, num, exp)
    generator.generate()


if __name__ == '__main__':
    main()
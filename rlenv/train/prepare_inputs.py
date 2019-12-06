import argparse
import os
from compress_pickle import load
import pandas as pd
from constants import PARTITIONS, REWARDS_DIR
from rlenv.Recorder import LSTG, REWARD


def accumulate_rewards(reward_dir, chunks):
    rewards = []
    for chunk in chunks:
        chunk_dir = '{}/{}/'.format(reward_dir, chunk)
        pieces = os.listdir(chunk_dir)
        for piece in pieces:
            piece_path = '{}{}.gz'.format(chunk_dir, piece)
            curr_rewards = load(piece_path)
            rewards.append(curr_rewards)
    rewards = pd.concat(rewards, ignore_index=True)
    rewards = rewards[[LSTG, REWARD]].groupby(LSTG).mean()
    return rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', required=True,
                        help='partition to chunk: {}'.format(PARTITIONS))
    part = parser.parse_args().part
    base_dir = '{}{}/'.format(REWARDS_DIR, part)
    reward_dir = '{}rewards'.format(base_dir)
    chunks = [path for path in os.listdir(reward_dir) if os.path.isdir(path)]
    rewards = accumulate_rewards(reward_dir, chunks)
    x_

if __name__ == '__main__':
    main()

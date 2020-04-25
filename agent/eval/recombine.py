import argparse
import numpy as np
import pandas as pd
from compress_pickle import load
from torch.utils.tensorboard import SummaryWriter
from constants import BYR_PREFIX, SLR_PREFIX, RL_LOG_DIR, SIM_CHUNKS


def gen_tag(s):
    return 'ValidationReturn/{}'.format(s)


def main():
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--byr', action='store_true')
    parser.add_argument('--run', type=str, required=True)
    args = parser.parse_args()
    byr, run = args.byr, args.run
    role = BYR_PREFIX if byr else SLR_PREFIX

    # find all runs
    parent_dir = RL_LOG_DIR + '{}/'.format(role)
    params = load(parent_dir + 'runs.pkl').loc[run]

    # tensorboard log
    run_dir = parent_dir + 'run_{}/'.format(run)
    writer = SummaryWriter(run_dir)

    # loop over iterations of model
    iterations = int(params['batch_count'])
    for itr in range(iterations):
        # load chunks
        rewards = list()
        for i in range(SIM_CHUNKS):
            path = run_dir + 'rewards/{}/{}.gz'.format(itr, i)
            rewards.append(load(path))

        # single series
        rewards = pd.concat(rewards)

        # save to tensorboard
        writer.add_scalar(gen_tag('Average'), np.mean(rewards), itr)
        writer.add_scalar(gen_tag('Max'), np.max(rewards), itr)
        writer.add_scalar(gen_tag('Median'), np.median(rewards), itr)
        writer.add_scalar(gen_tag('Min'), np.min(rewards), itr)
        writer.add_scalar(gen_tag('Std'), np.std(rewards), itr)
        writer.add_histogram(gen_tag('Histogram'), rewards, itr)


if __name__ == '__main__':
    main()

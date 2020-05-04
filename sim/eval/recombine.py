import argparse
import os
import numpy as np
import pandas as pd
from compress_pickle import load
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_multiplexer import \
    EventMultiplexer
from constants import BYR_PREFIX, SLR_PREFIX, RL_LOG_DIR, SIM_CHUNKS

SCALARS = {'Average': np.mean,
           'Max': np.max,
           'Median': np.median,
           'Min': np.min,
           'Std': np.std}


def gen_tag(s):
    return 'ValidationReturn/{}'.format(s)


def main():
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--byr', action='store_true')
    role = BYR_PREFIX if parser.parse_args().byr else SLR_PREFIX

    # find all runs
    parent_dir = RL_LOG_DIR + '{}/'.format(role)
    params = load(parent_dir + 'runs.pkl')
    runs = params.index

    # loop over runs
    for run in runs:
        # run directory
        run_dir = parent_dir + 'run_{}/'.format(run)

        # number of iterations
        iterations = int(params.loc[run, 'batch_count'])

        # TODO: continue if writer is already populated
        em = EventMultiplexer().AddRunsFromDirectory(run_dir).Reload()
        try:
            _ = em.Scalars('.', gen_tag('Average'))
            print('{}: already processed.'.format(run))
            continue
        except KeyError:
            pass

        # tensorboard log
        writer = SummaryWriter(run_dir)

        # loop over iterations of model
        rewards = list()
        for itr in range(iterations):
            # load chunks
            elem = list()
            for i in range(1, SIM_CHUNKS+1):
                path = run_dir + 'itr/{}/rewards/{}.gz'.format(itr, i)
                if os.path.isfile(path):
                    elem.append(load(path))
                else:
                    print('{}: chunks not finished'.format(run))
                    continue

            # single series
            rewards.append(pd.concat(elem))

        # save to tensorboard
        if len(rewards) == iterations:
            print('{}: writing to tensorboard'.format(run))
            for itr in range(iterations):
                for k, v in SCALARS.items():
                    writer.add_scalar(gen_tag(k), v(rewards), itr)
                    writer.add_histogram(gen_tag('ln(reward)'),
                                         np.log(rewards),
                                         itr)


if __name__ == '__main__':
    main()

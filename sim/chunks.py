import os
import argparse
from time import process_time
import numpy as np
import pandas as pd
import torch
from torch.distributions.categorical import Categorical
from rlenv.Composer import Composer
from sim.arrivals import ArrivalSimulator, ArrivalInterface, ArrivalQueryStrategy
from utils import topickle, load_file, load_featnames, load_model
from constants import NUM_CHUNKS, ARRIVAL_SIMS, INTERVAL
from paths import PARTS_DIR
from featnames import LOOKUP, X_LSTG, ARRIVALS, START_PRICE, START_TIME, \
    DEC_PRICE, ACC_PRICE, AGENT_PARTITIONS, FIRST_ARRIVAL_MODEL

LOOKUP_COLS = [START_PRICE, DEC_PRICE, ACC_PRICE, START_TIME]


def get_first_arrivals(x=None):
    net = load_model(FIRST_ARRIVAL_MODEL)
    x = {k: torch.from_numpy(v.values) for k, v in x.items()}
    dist = Categorical(logits=net(x))
    intervals = dist.sample(torch.Size(torch.tensor((ARRIVAL_SIMS,)))).numpy()
    sec = (intervals + np.random.uniform(size=np.shape(intervals))) * INTERVAL
    return sec.astype(np.uint32).transpose()


def main():
    # command line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str, choices=AGENT_PARTITIONS)
    parser.add_argument('--num', type=int,
                        choices=range(1, NUM_CHUNKS + 1))
    args = parser.parse_args()
    part, num = args.part, args.num - 1

    # create directory
    chunk_dir = PARTS_DIR + '{}/chunks/'.format(part)
    if not os.path.isdir(chunk_dir):
        os.mkdir(chunk_dir)

    path = chunk_dir + '{}.pkl'.format(num)
    if os.path.isfile(path):
        print('{} chunk #{} already exists'.format(part, num))
        exit(0)
    print('Saving {} chunk #{}'.format(part, num))

    # lookup and listing features
    lookup = load_file(part, LOOKUP)
    x_lstg = load_file(part, X_LSTG)
    featnames = load_featnames(X_LSTG)
    x_lstg = {k: pd.DataFrame(v, columns=featnames[k], index=lookup.index)
              for k, v in x_lstg.items()}

    # listings for chunk
    lstgs = [lookup.index[k] for k in range(len(lookup.index))
             if k % NUM_CHUNKS == num]
    lookup = lookup.loc[lstgs, LOOKUP_COLS]
    x_lstg = {k: v.loc[lstgs] for k, v in x_lstg.items()}
    for k, v in x_lstg.items():
        assert v.isna().sum().sum() == 0
        assert np.all(lstgs == v.index)

    # initialize arrival simulator
    composer = Composer()
    qs = ArrivalQueryStrategy(arrival=ArrivalInterface())
    simulator = ArrivalSimulator(composer=composer, query_strategy=qs)

    # sample first arrivals
    first_arrivals = get_first_arrivals(x=x_lstg)

    # loop over listings
    arrivals = {}
    for i, lstg in enumerate(lstgs):
        t0 = process_time()
        x_i = {k: v.loc[lstg].values for k, v in x_lstg.items()}
        simulator.set_lstg(x_lstg=x_i,
                           first_arrivals=first_arrivals[i, :],
                           start_time=lookup.loc[lstg, START_TIME])
        arrivals[lstg] = []
        for _ in range(ARRIVAL_SIMS):
            tups = simulator.simulate_arrivals()
            arrivals[lstg].append(tups)

        t1 = process_time()
        print('Listing {0:d} of {1:d}: {2:.1f}sec'.format(i+1, len(lstgs), t1-t0))

    # save
    chunk = {LOOKUP: lookup, X_LSTG: x_lstg, ARRIVALS: arrivals}
    topickle(chunk, chunk_dir + '{}.pkl'.format(num))


if __name__ == '__main__':
    main()

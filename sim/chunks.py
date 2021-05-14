import os
import argparse
from time import process_time
import numpy as np
import pandas as pd
from rlenv.Composer import Composer
from sim.arrivals import ArrivalInterface, ArrivalQueryStrategy, ArrivalSimulator
from utils import topickle, load_file, load_featnames
from constants import PARTS_DIR, NUM_CHUNKS, ARRIVAL_SIMS
from featnames import LOOKUP, X_LSTG, ARRIVALS, META, START_PRICE, START_TIME, \
    DEC_PRICE, ACC_PRICE, AGENT_PARTITIONS

LOOKUP_COLS = [META, START_PRICE, DEC_PRICE, ACC_PRICE, START_TIME]


def main():
    # command line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str, choices=AGENT_PARTITIONS)
    parser.add_argument('--num', type=int,
                        choices=range(1, NUM_CHUNKS + 1))
    args = parser.parse_args()
    part, num = args.part, args.num - 1
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
        assert np.all(lookup.index == v.index)

    # arrival interface and composer
    qs = ArrivalQueryStrategy(arrival=ArrivalInterface())
    composer = Composer()

    # loop over listings
    arrivals = {}
    for i, lstg in enumerate(lstgs):
        t0 = process_time()
        x_i = {k: v.loc[lstg].values for k, v in x_lstg.items()}
        simulator = ArrivalSimulator(start_time=lookup.loc[lstg, START_TIME],
                                     composer=composer,
                                     query_strategy=qs,
                                     x_lstg=x_i)
        arrivals[lstg] = []
        for _ in range(ARRIVAL_SIMS):
            arrivals[lstg].append(simulator.simulate_arrivals())

        t1 = process_time()
        print('Listing {0:d} of {1:d}: {2:.1f}sec'.format(i+1, len(lstgs), t1-t0))

    # create directory
    out_dir = PARTS_DIR + '{}/chunks/'.format(part)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # save
    chunk = {LOOKUP: lookup, X_LSTG: x_lstg, ARRIVALS: arrivals}
    topickle(chunk, out_dir + '{}.pkl'.format(num))


if __name__ == '__main__':
    main()

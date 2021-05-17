import os
import argparse
from time import process_time
import numpy as np
import pandas as pd
import torch
from rlenv.Composer import Composer
from rlenv.util import sample_categorical
from sim.arrivals import ArrivalSimulator
from utils import topickle, load_file, load_featnames, load_model
from constants import PARTS_DIR, NUM_CHUNKS, ARRIVAL_SIMS, INTERVAL_ARRIVAL
from featnames import LOOKUP, X_LSTG, ARRIVALS, META, START_PRICE, START_TIME, \
    DEC_PRICE, ACC_PRICE, AGENT_PARTITIONS, INTERARRIVAL_MODEL, FIRST_ARRIVAL_MODEL

LOOKUP_COLS = [META, START_PRICE, DEC_PRICE, ACC_PRICE, START_TIME]


class ArrivalQueryStrategy:
    def __init__(self, arrival=None):
        self.arrival = arrival

    def get_first_arrival(self, **kwargs):
        interval = self.arrival.first_arrival(logits=kwargs['logits'])
        return self._seconds_from_interval(interval)

    def get_inter_arrival(self, **kwargs):
        interval = self.arrival.inter_arrival(input_dict=kwargs['input_dict'])
        return self._seconds_from_interval(interval)

    @staticmethod
    def _seconds_from_interval(interval=None):
        return int((interval + np.random.uniform()) * INTERVAL_ARRIVAL)


class ArrivalInterface:
    def __init__(self):
        self.interarrival_model = load_model(INTERARRIVAL_MODEL)

    @staticmethod
    def first_arrival(logits=None):
        sample = sample_categorical(logits=logits)
        return sample

    def inter_arrival(self, input_dict=None):
        logits = self.interarrival_model(input_dict).cpu().squeeze()
        sample = sample_categorical(logits=logits)
        return sample


def first_arrival_pdf(x=None):
    net = load_model(FIRST_ARRIVAL_MODEL)
    x = {k: torch.from_numpy(v.values) for k, v in x.items()}
    theta = net(x)
    return theta


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
        assert np.all(lstgs == v.index)

    # initialize arrival simulator
    composer = Composer()
    qs = ArrivalQueryStrategy(arrival=ArrivalInterface())
    simulator = ArrivalSimulator(composer=composer, query_strategy=qs)

    # first arrival pdf
    logits0 = first_arrival_pdf(x=x_lstg)

    # loop over listings
    arrivals = {}
    for i, lstg in enumerate(lstgs):
        t0 = process_time()
        x_i = {k: v.loc[lstg].values for k, v in x_lstg.items()}
        simulator.set_lstg(x_lstg=x_i,
                           logits0=logits0[i, :],
                           start_time=lookup.loc[lstg, START_TIME])
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

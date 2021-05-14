import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import log_softmax
from rlenv.Composer import Composer
from rlenv.Sources import Sources
from rlenv.const import CLOCK_MAP
from rlenv.util import get_clock_feats, sample_categorical
from utils import topickle, load_file, load_featnames, load_model, get_days_since_lstg
from constants import PARTS_DIR, NUM_CHUNKS, INTERVAL_ARRIVAL, INTERVAL_CT_ARRIVAL, \
    MAX_DELAY_ARRIVAL, DAY, ARRIVAL_SIMS
from featnames import LOOKUP, X_LSTG, ARRIVALS, META, START_PRICE, START_TIME, \
    DEC_PRICE, ACC_PRICE, FIRST_ARRIVAL_MODEL, INTERARRIVAL_MODEL, \
    AGENT_PARTITIONS, LSTG, DAYS_SINCE_LAST, THREAD_COUNT, DAYS_SINCE_LSTG

LOOKUP_COLS = [META, START_PRICE, DEC_PRICE, ACC_PRICE, START_TIME]


class ArrivalInterface:
    def __init__(self):
        self.interarrival_model = load_model(INTERARRIVAL_MODEL)

    def inter_arrival(self, input_dict=None):
        logits = self.interarrival_model(input_dict).cpu().squeeze()
        sample = sample_categorical(logits=logits)
        return sample


class ArrivalSources(Sources):
    def __init__(self, x_lstg=None):
        super().__init__(x_lstg=x_lstg)
        self.source_dict[DAYS_SINCE_LAST] = 0.0
        self.source_dict[THREAD_COUNT] = 0.0

    def update_arrival(self, clock_feats=None, days_since_lstg=None,
                       thread_count=None, days_since_last=None):
        self.source_dict[THREAD_COUNT] = thread_count
        self.source_dict[CLOCK_MAP] = clock_feats
        self.source_dict[DAYS_SINCE_LSTG] = days_since_lstg
        self.source_dict[DAYS_SINCE_LAST] = days_since_last


def get_p_arrival(x_lstg=None):
    net = load_model(FIRST_ARRIVAL_MODEL)
    input_dict = {k: torch.from_numpy(v.values) for k, v in x_lstg.items()}
    theta = net(input_dict)
    p = np.exp(log_softmax(theta, dim=-1).numpy())
    p = pd.DataFrame(p, index=x_lstg[LSTG].index, dtype='float32')
    totals = p.sum(axis=1)
    assert all(abs(totals - 1) < 1e-6)
    return p


def get_seconds(p=None):
    interval = np.random.choice(INTERVAL_CT_ARRIVAL + 1, size=ARRIVAL_SIMS, p=p.values)
    noise = np.random.uniform(size=ARRIVAL_SIMS)
    seconds = ((interval + noise) * INTERVAL_ARRIVAL).astype(np.int64)
    return seconds


def update_sources(sources=None, start_time=None, arrival_time=None,
                   thread_count=None, last_arrival_time=None):
    clock_feats = get_clock_feats(arrival_time)
    days_since_lstg = get_days_since_lstg(lstg_start=start_time, time=arrival_time)
    days_since_last = (arrival_time - last_arrival_time) / DAY
    update_args = dict(clock_feats=clock_feats,
                       days_since_lstg=days_since_lstg,
                       days_since_last=days_since_last,
                       thread_count=thread_count)
    sources.update_arrival(**update_args)


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

    # first arrival probabilities
    p1 = get_p_arrival(x_lstg)

    # arrival interface and composer
    interface = ArrivalInterface()
    composer = Composer()

    # loop over listings
    arrivals = {}
    for lstg in lstgs:
        arrivals[lstg] = []
        start_time = lookup.loc[lstg, START_TIME]
        end_time = start_time + MAX_DELAY_ARRIVAL
        first_arrivals = start_time + get_seconds(p1.loc[lstg])
        x_i = {k: v.loc[lstg].values for k, v in x_lstg.items()}
        for sim in range(10):
            times = []
            thread_count = 0
            last_arrival_time = start_time
            time = int(first_arrivals[sim])
            sources = ArrivalSources(x_lstg=x_i)
            while time < end_time:
                times.append(time)
                thread_count += 1
                update_sources(sources=sources,
                               start_time=start_time,
                               arrival_time=time,
                               thread_count=thread_count,
                               last_arrival_time=last_arrival_time)
                last_arrival_time = time
                input_dict = composer.build_input_dict(model_name=INTERARRIVAL_MODEL,
                                                       sources=sources(),
                                                       turn=None)
                interval = interface.inter_arrival(input_dict)
                seconds = int((interval + np.random.uniform()) * INTERVAL_ARRIVAL)
                time = min(time + seconds, end_time)
            arrivals[lstg].append(times)

    # create directory
    out_dir = PARTS_DIR + '{}/chunks/'.format(part)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # save
    chunk = {LOOKUP: lookup, X_LSTG: x_lstg, ARRIVALS: arrivals}
    topickle(chunk, out_dir + '{}.pkl'.format(num))


if __name__ == '__main__':
    main()

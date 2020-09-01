import argparse
from datetime import datetime as dt
import pandas as pd
from processing.b_feats.util import create_events, get_all_cat_feats
from processing.util import load_feats
from utils import unpickle, topickle, run_func_on_chunks
from constants import FEATS_DIR
from featnames import META, LEAF, SLR, LSTG


def create_feats(data=None, name=None):
    events = create_events(data=data, levels=[name])  # set levels
    feats = get_all_cat_feats(events, [name])  # categorical features
    assert not feats.isna().any().any()
    return feats


def process_subset(data=None, name=None, meta=None, chunk=None):
    start = dt.now()
    val = meta[chunk]
    lstgs = data['listings'].loc[data['listings'][META] == val].index
    for k, v in data.items():
        if len(v.index.names) == 1:
            data[k] = v.reindex(index=lstgs)
        else:
            data[k] = v.reindex(index=lstgs, level=LSTG)
    feats = create_feats(data=data, name=name)
    sec = (dt.now() - start).total_seconds()
    print('{} {}: {} listings, {} seconds'.format(META, val, len(lstgs), sec))
    return feats


def process_chunk(chunk=None):
    start = dt.now()
    data = unpickle(FEATS_DIR + 'chunks/{}.pkl'.format(chunk))
    feats = create_feats(data=data, name=SLR)
    sec = (dt.now() - start).total_seconds()
    print('Chunk {}: {} listings, {} seconds'.format(chunk, len(feats), sec))
    return feats


def main():
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',
                        choices=[SLR, META, LEAF],
                        required=True)
    name = parser.parse_args().name
    print('Creating feats at {} level'.format(name))

    if name == SLR:
        res = run_func_on_chunks(f=process_chunk, func_kwargs=dict())

    else:
        # load data
        data = dict()
        for level in ['listings', 'threads', 'offers']:
            data[level] = load_feats(level)

        # parallel processing
        meta = data['listings'][META].unique()
        print('{} unique meta categories'.format(len(meta)))
        kwargs = dict(data=data, name=name, meta=meta)
        res = run_func_on_chunks(f=process_subset,
                                 func_kwargs=kwargs,
                                 num_chunks=len(meta))

    # concatenate dataframes
    feats = pd.concat(res).sort_index()

    # save
    topickle(feats, FEATS_DIR + '{}.pkl'.format(name))


if __name__ == "__main__":
    main()

import argparse
import os
from datetime import datetime as dt
import pandas as pd
from processing.b_feats.util import create_events, get_all_cat_feats
from utils import unpickle, topickle, run_func_on_chunks
from constants import FEATS_DIR
from featnames import META, LEAF, SLR


def create_feats(data=None, name=None):
    events = create_events(data=data, levels=[name])  # set levels
    feats = get_all_cat_feats(events, [name])  # categorical features
    assert not feats.isna().any().any()
    return feats


def process_chunk(chunk=None, name=None):
    start = dt.now()
    path = FEATS_DIR + 'chunks/{}{}.pkl'.format(META if name != SLR else '', chunk)
    data = unpickle(path)
    feats = create_feats(data=data, name=name)
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

    kw = dict(f=process_chunk, func_kwargs=dict(name=name))
    if name != SLR:
        chunks_dir = FEATS_DIR + 'chunks/'
        kw['num_chunks'] = len([name for name in os.listdir(chunks_dir)
                                if name.startswith(META)])
    res = run_func_on_chunks(**kw)

    # concatenate dataframes
    feats = pd.concat(res).sort_index()

    # save
    topickle(feats, FEATS_DIR + '{}.pkl'.format(name))


if __name__ == "__main__":
    main()

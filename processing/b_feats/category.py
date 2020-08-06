import argparse
from compress_pickle import load, dump
from datetime import datetime as dt
import pandas as pd
from processing.b_feats.util import create_events, get_all_cat_feats
from processing.util import load_feats
from utils import run_func_on_chunks
from constants import FEATS_DIR, NUM_FEATS_CHUNKS
from featnames import META, LEAF, SLR


def create_feats(data=None, name=None):
    # set levels for hierarchical feats
    events = create_events(data=data, levels=[name])
    # categorical features
    start = dt.now()
    feats = get_all_cat_feats(events, [name])
    assert not feats.isna().any().any()
    print('{} seconds'.format((dt.now() - start).total_seconds()))
    return feats


def process_chunk(chunk=None):
    chunk = load(FEATS_DIR + 'chunks/{}.gz'.format(chunk))
    return create_feats(data=chunk, name=SLR)


def main():
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',
                        choices=[SLR, META, LEAF],
                        required=True)
    name = parser.parse_args().name

    if name == SLR:
        res = run_func_on_chunks(
            num_chunks=NUM_FEATS_CHUNKS,
            f=process_chunk,
            func_kwargs=dict()
        )
        feats = pd.concat(res).sort_index()
    else:
        data = dict()
        for level in ['listings', 'threads', 'offers']:
            data[level] = load_feats(name)
        feats = create_feats(data=data, name=name)

    # save
    dump(feats, FEATS_DIR + '{}.gz'.format(name))


if __name__ == "__main__":
    main()

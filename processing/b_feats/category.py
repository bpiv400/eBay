import argparse
from compress_pickle import dump
from datetime import datetime as dt
from processing.b_feats.util import create_events, get_all_cat_feats
from processing.util import load_feats
from constants import FEATS_DIR
from featnames import SLR, META, LEAF


def main():
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, choices=[SLR, META, LEAF],
                        required=True)
    name = parser.parse_args().name

    # load files
    data = {}
    for level in ['listings', 'threads', 'offers']:
        data[level] = load_feats(level)

    # set levels for hierarchical feats
    events = create_events(data=data, levels=[name])

    # categorical features
    start = dt.now()
    feats = get_all_cat_feats(events, [name])
    assert not feats.isna().any().any()
    print('{} seconds'.format((dt.now() - start).total_seconds()))

    # save
    dump(feats, FEATS_DIR + '{}.gz'.format(name))


if __name__ == "__main__":
    main()

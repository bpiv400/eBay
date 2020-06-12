import argparse
from compress_pickle import dump, load
from datetime import datetime as dt
import processing.b_feats.util as util
from constants import PARTITIONS, PARTS_DIR, CLEAN_DIR


def main():
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--part',
                        type=str,
                        choices=PARTITIONS,
                        required=True)
    parser.add_argument('--name',
                        type=str,
                        choices=['slr', 'meta', 'leaf'],
                        required=True)
    args = parser.parse_args()
    part, name = args.part, args.name
    print('{}/{}'.format(part, name))

    # listing ids
    idx = load(PARTS_DIR + 'partitions.pkl')[part]

    # load files
    offers = load(CLEAN_DIR + 'offers.pkl').reindex(
        index=idx, level='lstg')
    threads = load(CLEAN_DIR + 'threads.pkl').reindex(
        index=idx, level='lstg')
    listings = load(CLEAN_DIR + 'listings.pkl').reindex(index=idx)

    # set levels for hierarchical feats
    print('Creating offer events...')
    events = util.create_events(listings, threads, offers, [name])

    # categorical features
    print('Creating categorical features...')
    start = dt.now()
    feats = util.get_all_cat_feats(events, [name])
    assert not feats.isna().any().any()
    print('{} seconds'.format((dt.now() - start).total_seconds()))

    # save
    dump(feats, PARTS_DIR + '{}/{}.gz'.format(part, name))


if __name__ == "__main__":
    main()

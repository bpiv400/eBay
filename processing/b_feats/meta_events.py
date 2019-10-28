"""
Creates events object indexed by meta, leaf, cndtn, lstg, thread, index
for creation of categorical features
"""
import argparse
from compress_pickle import dump, load
import processing.b_feats.util as util
from constants import FEATS_DIR, CHUNKS_DIR


def main():
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num
    # load data
    print('Loading data')
    d = load(CHUNKS_DIR + 'm%d' % num + '.gz')
    L, T, O = [d[k] for k in ['listings', 'threads', 'offers']]

    # set levels for hierarchical time feats
    levels = ['meta', 'leaf', 'cndtn']

    # create events dataframe
    print('Creating offer events.')
    events = util.create_events(L, T, O, levels)

    print('Saving')
    events_file = '{}m{}_events.gz'.format(FEATS_DIR, num)
    dump(events, events_file)


if __name__ == '__main__':
    main()

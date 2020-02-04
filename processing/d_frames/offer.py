from compress_pickle import load, dump
import numpy as np, pandas as pd
from processing.processing_utils import input_partition, \
    collect_date_clock_feats, get_days_delay, get_norm, get_con
from processing.d_frames.frames_utils import get_partition, load_frames
from processing.processing_consts import CLEAN_DIR
from constants import START, PARTS_DIR, SLR_PREFIX, IDX
from featnames import DAYS, DELAY, CON, NORM, SPLIT, MSG, REJECT, AUTO, EXP
from utils import is_split


def get_x_offer(start_price, events, tf):
    # initialize output dataframe
    df = pd.DataFrame(index=events.index).sort_index()

    # clock features
    df = df.join(collect_date_clock_feats(events.clock))

    # differenced time feats
    df = df.join(tf.reindex(index=df.index, fill_value=0))

    # delay features
    df[DAYS], df[DELAY] = get_days_delay(events.clock.unstack())

    # auto and exp are functions of delay
    df[AUTO] = (df[DELAY] == 0) & df.index.isin(IDX[SLR_PREFIX], level='index')
    df[EXP] = (df[DELAY] == 1) | events.censored

    # concession
    df[CON] = get_con(events.price.unstack(), start_price)

    # reject, total concession, and split are functions of con
    df[REJECT] = df[CON] == 0
    df[NORM] = get_norm(df[CON])
    df[SPLIT] = df[CON].apply(is_split)

    # message indicator is last
    df[MSG] = events.message

    return df


def main():
    # partition and corresponding indices
    part = input_partition()
    idx, path = get_partition(part)
    print('{}/x_offer'.format(part))

    # load other data
    start_price = load(PARTS_DIR + '%s/lookup.gz' % part).start_price
    events = load(CLEAN_DIR + 'offers.pkl').reindex(index=idx, level='lstg')
    tf = load_frames('tf_offer').reindex(index=idx, level='lstg')

    # offer features
    x_offer = get_x_offer(start_price, events, tf)
    dump(x_offer, path('x_offer'))

    # offer timestamps
    dump(events.clock, path('clock'))


if __name__ == "__main__":
    main()
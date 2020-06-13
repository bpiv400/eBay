from compress_pickle import load, dump
import pandas as pd
from processing.utils import collect_date_clock_feats, \
    get_days_delay, get_norm, get_con
from util import input_partition, load_file, is_split
from constants import PARTS_DIR, SLR_PREFIX, IDX, CLEAN_DIR
from featnames import DAYS, DELAY, CON, NORM, SPLIT, MSG, REJECT, \
    AUTO, EXP, TIME_FEATS


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

    # set time feats to 0 for censored observations
    df.loc[(df[DELAY] < 1) & df[EXP], TIME_FEATS] = 0.0

    return df


def main():
    # partition and corresponding indices
    part = input_partition()
    print('{}/x_offer'.format(part))

    # lstg index
    idx = load_file(part, 'lookup').index

    # load other data
    start_price = load(PARTS_DIR + '%s/lookup.gz' % part).start_price
    events = load(CLEAN_DIR + 'offers.pkl').reindex(index=idx, level='lstg')
    tf = load(PARTS_DIR + 'tf.gz').reindex(index=idx, level='lstg')

    # offer features
    x_offer = get_x_offer(start_price, events, tf)
    dump(x_offer, PARTS_DIR + '{}/x_offer.gz'.format(part))

    # offer timestamps
    dump(events.clock, PARTS_DIR + '{}/clock.gz'.format(part))


if __name__ == "__main__":
    main()

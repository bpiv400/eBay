import pandas as pd
from processing.util import collect_date_clock_feats, \
    get_days_delay, get_lstgs
from utils import topickle, is_split, input_partition, load_feats
from constants import PARTS_DIR, IDX
from featnames import DAYS, DELAY, CON, NORM, SPLIT, MSG, REJECT, \
    AUTO, EXP, START_PRICE, SLR, CLOCK, INDEX


def get_x_offer(start_price, offers, tf):
    # initialize output dataframe
    df = pd.DataFrame(index=offers.index).sort_index()

    # clock features
    df = df.join(collect_date_clock_feats(offers[CLOCK]))

    # differenced time feats
    df = df.join(tf.reindex(index=df.index, fill_value=0))

    # delay features
    df[DAYS], df[DELAY] = get_days_delay(offers[CLOCK].unstack())

    # auto and exp are functions of delay
    df[AUTO] = (df[DELAY] == 0) & df.index.isin(IDX[SLR], level=INDEX)
    df[EXP] = df[DELAY] == 1

    # concession
    df.loc[[CON, REJECT, NORM]] = offers[[CON, REJECT, NORM]]
    df[SPLIT] = df[CON].apply(is_split)

    # message indicator is last
    df[MSG] = offers.message

    # error checking
    assert all(df.loc[df[EXP], REJECT])

    return df


def create_x_offer(lstgs=None):
    # load data
    start_price = load_feats('listings', lstgs=lstgs)[START_PRICE]
    offers = load_feats('offers', lstgs=lstgs)
    tf = load_feats('tf', lstgs=lstgs)

    # offer features
    x_offer = get_x_offer(start_price, offers, tf)

    return x_offer, offers.clock


def main():
    part = input_partition()
    print('{}/x_offer'.format(part))

    x_offer, clock = create_x_offer(lstgs=get_lstgs(part))
    topickle(x_offer, PARTS_DIR + '{}/x_offer.pkl'.format(part))
    topickle(clock, PARTS_DIR + '{}/clock.pkl'.format(part))


if __name__ == "__main__":
    main()

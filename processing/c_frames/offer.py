from compress_pickle import dump
import pandas as pd
from processing.util import collect_date_clock_feats, \
    get_days_delay, get_norm, get_con, get_lstgs, load_feats
from utils import is_split, input_partition
from constants import MODEL_PARTS_DIR, SLR, IDX, TRAIN_MODELS, VALIDATION, TEST
from featnames import DAYS, DELAY, CON, NORM, SPLIT, MSG, REJECT, \
    AUTO, EXP, TIME_FEATS, START_PRICE


def get_x_offer(start_price, offers, tf):
    # initialize output dataframe
    df = pd.DataFrame(index=offers.index).sort_index()

    # clock features
    df = df.join(collect_date_clock_feats(offers.clock))

    # differenced time feats
    df = df.join(tf.reindex(index=df.index, fill_value=0))

    # delay features
    df[DAYS], df[DELAY] = get_days_delay(offers.clock.unstack())

    # auto and exp are functions of delay
    df[AUTO] = (df[DELAY] == 0) & df.index.isin(IDX[SLR], level='index')
    df[EXP] = (df[DELAY] == 1) | offers.censored

    # concession
    df[CON] = get_con(offers.price.unstack(), start_price)

    # reject, total concession, and split are functions of con
    df[REJECT] = df[CON] == 0
    df[NORM] = get_norm(df[CON])
    df[SPLIT] = df[CON].apply(is_split)

    # message indicator is last
    df[MSG] = offers.message

    # set time feats to 0 for censored observations
    df.loc[(df[DELAY] < 1) & df[EXP], TIME_FEATS] = 0.0

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
    assert part in [TRAIN_MODELS, VALIDATION, TEST]
    print('{}/x_offer'.format(part))

    x_offer, clock = create_x_offer(lstgs=get_lstgs(part))
    dump(x_offer, MODEL_PARTS_DIR + '{}/x_offer.gz'.format(part))
    dump(clock, MODEL_PARTS_DIR + '{}/clock.gz'.format(part))


if __name__ == "__main__":
    main()

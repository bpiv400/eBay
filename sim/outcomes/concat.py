import pandas as pd
from compress_pickle import load, dump
from processing.util import collect_date_clock_feats, \
    get_days_delay, get_norm
from utils import is_split, input_partition, load_file
from constants import SIM_CHUNKS, IDX, SLR_PREFIX, MONTH, PARTS_DIR
from featnames import DAYS, DELAY, CON, SPLIT, NORM, REJECT, AUTO, EXP, \
    CENSORED, CLOCK_FEATS, TIME_FEATS, OUTCOME_FEATS, MONTHS_SINCE_LSTG, \
    BYR_HIST


def diff_tf(df):
    # pull out time feats
    tf = df[TIME_FEATS].astype('float64')
    df.drop(TIME_FEATS, axis=1, inplace=True)
    # for each feat, unstack, difference, and stack
    for c in TIME_FEATS:
        wide = tf[c].unstack()
        first = wide[[1]].stack()
        diff = wide.diff(axis=1).stack()
        df[c] = pd.concat([first, diff], axis=0).sort_index()
        assert df[c].isna().sum() == 0
    # censored feats to 0
    df.loc[df[CENSORED], TIME_FEATS] = 0
    return df


def process_sim_offers(df, end_time):
    # difference time features
    df = diff_tf(df)
    # censor timestamps
    timestamps = df.clock.to_frame().join(end_time.rename('end'))
    timestamps = timestamps.reorder_levels(df.index.names)
    clock = timestamps.min(axis=1)
    # clock features
    df = df.join(collect_date_clock_feats(clock))
    # days and delay
    df[DAYS], df[DELAY] = get_days_delay(clock.unstack())
    # concession as a decimal
    df.loc[:, CON] /= 100
    # indicator for split
    df[SPLIT] = df[CON].apply(lambda x: is_split(x))
    # total concession
    df[NORM] = get_norm(df[CON])
    # reject auto and exp are last
    df[REJECT] = df[CON] == 0
    df[AUTO] = (df[DELAY] == 0) & df.index.isin(IDX[SLR_PREFIX], level='index')
    df[EXP] = (df[DELAY] == 1) | df[CENSORED]
    # select and sort features
    df = df.loc[:, CLOCK_FEATS + TIME_FEATS + OUTCOME_FEATS]
    return df, clock


def process_sim_threads(df, start_time):
    # convert clock to months_since_lstg
    df = df.join(start_time)
    df[MONTHS_SINCE_LSTG] = (df.clock - df.start_time) / MONTH
    df = df.drop(['clock', 'start_time'], axis=1)
    # reorder columns to match observed
    df = df.loc[:, [MONTHS_SINCE_LSTG, BYR_HIST]]
    return df


def concat_sim_chunks(part):
    """
    Loops over simulations, concatenates dataframes.
    :param str part: name of partition.
    :return tuple (threads, offers): dataframes of threads and offers.
    """
    # collect chunks
    threads, offers = [], []
    for i in range(1, SIM_CHUNKS + 1):
        sim = load(PARTS_DIR + '{}/outcomes/{}.gz'.format(part, i))
        threads.append(sim['threads'])
        offers.append(sim['offers'])

    # concatenate
    threads = pd.concat(threads, axis=0).sort_index()
    offers = pd.concat(offers, axis=0).sort_index()

    return threads, offers


def clean_components(threads, offers, part):
    # output dictionary
    d = dict()

    # index of 'lstg' or ['lstg', 'sim']
    idx = threads.reset_index('thread', drop=True).index.unique()

    # end of listing
    sale_time = offers.loc[offers[CON] == 100, 'clock'].reset_index(
        level=['thread', 'index'], drop=True)
    d['lstg_end'] = sale_time.reindex(index=idx, fill_value=-1)
    no_sale = d['lstg_end'][d['lstg_end'] == -1].index
    d['lstg_end'].loc[no_sale] = d['lstg_end'].loc[no_sale] + MONTH - 1

    # conform to observed inputs
    lookup = load_file(part, 'lookup')
    d['x_thread'] = process_sim_threads(threads, lookup.start_time)
    d['x_offer'], d['clock'] = process_sim_offers(offers, d['lstg_end'])

    return d


def main():
    part = input_partition()

    # concatenate chunks
    threads, offers = concat_sim_chunks(part)

    # create output dataframes
    d = clean_components(threads, offers, part)

    # save
    for k, df in d.items():
        dump(df, PARTS_DIR + '{}/{}_sim.gz'.format(part, k))


if __name__ == '__main__':
    main()

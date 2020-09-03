import numpy as np
import pandas as pd
from inputs.util import save_files, get_x_thread, check_zero, get_ind_x
from processing.util import collect_date_clock_feats
from utils import input_partition, load_file
from constants import DAY, IDX, CON_MULTIPLIER, POLICY_BYR
from featnames import CON, THREAD_COUNT, LOOKUP, BYR, X_OFFER, X_THREAD, CLOCK, \
    INDEX, START_TIME, TIME_FEATS, OUTCOME_FEATS, MSG, REJECT, NORM, SPLIT, THREAD


def get_x_offer(offers, idx):
    x_offer = {}
    for i in range(1, max(IDX[BYR]) + 1):
        # offer features at turn i, and turn number
        offer = offers.xs(i, level=INDEX).reindex(
            index=idx, fill_value=0).astype('float32')
        turn = offer.index.get_level_values(level=INDEX)

        # msg is 0 for turns of focal player
        if i in IDX[BYR]:
            offer.loc[:, MSG] = 0.

        # all features are zero for future turns
        offer.loc[i > turn, :] = 0.

        # for current turn, post-delay features set to 0
        offer.loc[i == turn, [CON, REJECT, NORM, SPLIT]] = 0.

        # put in dictionary
        x_offer['offer{}'.format(i)] = offer

    # error checking
    check_zero(x_offer)

    return x_offer


def construct_x(idx=None, threads=None, offers=None):
    # thread features
    x_thread = get_x_thread(threads, idx, turn_indicators=True)
    x_thread.drop(THREAD_COUNT, axis=1, inplace=True)
    # initialize x with thread features
    x = {THREAD: x_thread}
    # offer features
    x.update(get_x_offer(offers, idx))
    # no nans
    for v in x.values():
        assert v.isna().sum().sum() == 0
    return x


def reshape_offers(offers=None, clock=None, idx0=None, idx1=None):
    df = offers.drop(TIME_FEATS, axis=1).reindex(index=idx1)
    arrivals = collect_date_clock_feats(clock.loc[idx0])
    outcomes = [False if df[c].dtype == 'bool' else 0. for c in OUTCOME_FEATS]
    arrivals[OUTCOME_FEATS] = outcomes
    df = pd.concat([df, arrivals], axis=0).sort_index()
    return df


def append_arrival_delays(clock=None, lstg_start=None):
    # buyer turns
    first_offer = clock.xs(1, level=INDEX, drop_level=False)
    later_offers = clock[clock.index.isin(range(2, 8), level=INDEX)]
    # days from listing start to thread start
    days = (first_offer - lstg_start).rename('day') // DAY
    first_offer = pd.concat([first_offer, days], axis=1).set_index(
        'day', append=True).squeeze()
    later_offers = later_offers.to_frame().assign(day=0).set_index(
        'day', append=True).squeeze()
    # expand by number of days
    ranges = days[days > 0].apply(range)
    wide = pd.DataFrame.from_records(ranges.values, index=ranges.index)
    s = wide.rename_axis('day', axis=1).stack().astype('int64')
    # create fake non-arrival timestamps
    ts_delay = s * DAY + lstg_start
    ts_delay += (np.random.uniform(size=len(ts_delay)) * DAY).astype('int64')
    # concatenate with offers
    ts_offer = pd.concat([first_offer, later_offers], axis=0).sort_index()
    ts = pd.concat([ts_delay, ts_offer], axis=0).sort_index()
    return ts, ts_delay.index, ts_offer.index


def process_inputs(part):
    # load dataframes
    offers = load_file(part, X_OFFER)
    threads = load_file(part, X_THREAD)
    clock = load_file(part, CLOCK)
    lookup = load_file(part, LOOKUP)

    # append arrival timestamps to clock
    clock, idx0, idx1 = append_arrival_delays(clock=clock,
                                              lstg_start=lookup[START_TIME])

    # reshape offers dataframe
    offers = reshape_offers(offers=offers,
                            clock=clock,
                            idx0=idx0,
                            idx1=idx1)
    assert np.all(offers.index == clock.index)

    # buyer index
    idx = clock[clock.index.isin(IDX[BYR], level=INDEX)].index

    # outcome
    y = (offers.loc[idx, CON] * CON_MULTIPLIER).astype('int8')

    # input feature dictionary
    x = construct_x(idx=idx, threads=threads, offers=offers)

    # indices for listing features
    idx_x = get_ind_x(lstgs=lookup.index, idx=idx)

    return {'y': y, 'x': x, 'idx_x': idx_x}


def main():
    # extract parameters from command line
    part = input_partition()
    print('{}/{}'.format(part, POLICY_BYR))

    # input dataframes, output processed dataframes
    d = process_inputs(part)

    # save various output files
    save_files(d, part, POLICY_BYR)


if __name__ == '__main__':
    main()

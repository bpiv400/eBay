import numpy as np
import pandas as pd
from inputs.util import save_files, get_x_thread, check_zero, get_ind_x
from processing.util import collect_date_clock_feats
from utils import load_file
from constants import DAY, IDX, CON_MULTIPLIER, POLICY_BYR, VALIDATION
from featnames import CON, THREAD_COUNT, LOOKUP, BYR, X_OFFER, X_THREAD, CLOCK, \
    INDEX, START_TIME, TIME_FEATS, OUTCOME_FEATS, MSG, REJECT, NORM, SPLIT, THREAD, \
    DAYS_SINCE_LSTG


def get_x_offer(offers, idx):
    # first turn
    first_offer = offers.xs(1, level=INDEX).astype('float32')
    offer1 = pd.DataFrame(index=idx).join(first_offer)
    offer1.index = offer1.index.reorder_levels(idx.names)
    offer1 = offer1.sort_index()
    turn = offer1.index.get_level_values(level=INDEX)
    offer1.loc[turn == 1, OUTCOME_FEATS] = 0.
    x_offer = {'offer1': offer1}

    # later turns
    for i in range(2, max(IDX[BYR]) + 1):
        # offer features at turn i, and turn number
        offer = offers.xs(i, level=INDEX).droplevel('day').reindex(
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
    check_zero(x_offer, byr_exp=False)

    return x_offer


def construct_x(idx=None, threads=None, clock=None, lookup=None, offers=None):
    # thread features
    x_thread = get_x_thread(threads, idx, turn_indicators=True)
    x_thread.drop(THREAD_COUNT, axis=1, inplace=True)
    days = (clock - lookup[START_TIME]) / DAY
    x_thread[DAYS_SINCE_LSTG] = np.minimum(x_thread[DAYS_SINCE_LSTG], days)
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


def append_arrival_delays(clock=None, lstg_start=None, delays=None):
    # buyer turns
    first_offer = clock.xs(1, level=INDEX, drop_level=False)
    later_offers = clock[clock.index.isin(range(2, 8), level=INDEX)]
    # days from listing start to thread start
    days = (first_offer - lstg_start).rename('day') // DAY
    first_offer = pd.concat([first_offer, days], axis=1).set_index(
        'day', append=True).squeeze()
    later_offers = later_offers.to_frame().join(days.droplevel(INDEX)).set_index(
        'day', append=True).squeeze()

    if delays is None:
        # create fake non-arrival timestamps
        ranges = days[days > 0].apply(range)
        wide = pd.DataFrame.from_records(ranges.values, index=ranges.index)
        s = wide.rename_axis('day', axis=1).stack().astype('int64')
        ts_delay = s * DAY + lstg_start
        ts_delay += (np.random.uniform(size=len(ts_delay)) * DAY).astype('int64')
    else:
        # match format and index
        ts_delay = delays + lstg_start

    # concatenate with offers
    ts_offer = pd.concat([first_offer, later_offers], axis=0).sort_index()
    ts = pd.concat([ts_delay, ts_offer], axis=0).sort_index()
    return ts, ts_delay.index, ts_offer.index


def process_byr_inputs(data=None):
    # append arrival timestamps to clock
    clock, idx0, idx1 = append_arrival_delays(clock=data[CLOCK],
                                              lstg_start=data[LOOKUP][START_TIME])

    # reshape offers dataframe
    offers = reshape_offers(offers=data[X_OFFER],
                            clock=clock,
                            idx0=idx0,
                            idx1=idx1)
    assert np.all(offers.index == clock.index)

    # buyer index
    idx = clock[clock.index.isin(IDX[BYR], level=INDEX)].index

    # outcome
    y = (offers.loc[idx, CON] * CON_MULTIPLIER).astype('int8')

    # input feature dictionary
    x = construct_x(idx=idx,
                    threads=data[X_THREAD],
                    clock=clock,
                    lookup=data[LOOKUP],
                    offers=offers)

    # indices for listing features
    idx_x = get_ind_x(lstgs=data[LOOKUP].index, idx=idx)

    return {'y': y, 'x': x, 'idx_x': idx_x}


def main():
    print('{}/{}'.format(VALIDATION, POLICY_BYR))

    data = {k: load_file(VALIDATION, k)
            for k in [LOOKUP, X_THREAD, X_OFFER, CLOCK]}

    # input dataframes, output processed dataframes
    d = process_byr_inputs(data)

    # save various output files
    save_files(d, VALIDATION, POLICY_BYR)


if __name__ == '__main__':
    main()

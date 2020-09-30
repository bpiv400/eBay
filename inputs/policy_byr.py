import numpy as np
import pandas as pd
from inputs.util import save_files, get_x_thread, check_zero, get_ind_x
from processing.util import collect_date_clock_feats
from utils import load_data
from constants import DAY, IDX, CON_MULTIPLIER, POLICY_BYR, VALIDATION
from featnames import CON, THREAD_COUNT, LOOKUP, BYR, X_OFFER, X_THREAD, CLOCK, \
    INDEX, START_TIME, TIME_FEATS, OUTCOME_FEATS, MSG, REJECT, NORM, SPLIT, THREAD, \
    DAYS_SINCE_LSTG, BYR_DELAYS, BYR_AGENT


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


def construct_x(idx=None, data=None):
    # thread features
    x_thread = get_x_thread(data[X_THREAD], idx, turn_indicators=True)
    x_thread.drop(THREAD_COUNT, axis=1, inplace=True)
    days = (data[CLOCK] - data[LOOKUP][START_TIME]) / DAY
    x_thread[DAYS_SINCE_LSTG] = np.minimum(x_thread[DAYS_SINCE_LSTG], days)
    # initialize x with thread features
    x = {THREAD: x_thread}
    # offer features
    x.update(get_x_offer(data[X_OFFER], idx))
    # no nans
    for v in x.values():
        assert v.isna().sum().sum() == 0
    return x


def reshape_offers(data=None, idx0=None, idx1=None):
    df = data[X_OFFER].drop(TIME_FEATS, axis=1).reindex(index=idx1)
    arrivals = collect_date_clock_feats(data[CLOCK].loc[idx0])
    outcomes = [False if df[c].dtype == 'bool' else 0. for c in OUTCOME_FEATS]
    arrivals[OUTCOME_FEATS] = outcomes
    data[X_OFFER] = pd.concat([df, arrivals], axis=0).sort_index()
    assert np.all(data[X_OFFER].index == data[CLOCK].index)


def append_arrival_delays(data=None):
    # buyer turns
    first_offer = data[CLOCK].xs(1, level=INDEX, drop_level=False)
    later_offers = data[CLOCK][data[CLOCK].index.isin(range(2, 8), level=INDEX)]

    # days from listing start to thread start
    days = (first_offer - data[LOOKUP][START_TIME]).rename('day') // DAY
    first_offer = pd.concat([first_offer, days], axis=1).set_index(
        'day', append=True).squeeze()
    later_offers = later_offers.to_frame().join(days.droplevel(INDEX)).set_index(
        'day', append=True).squeeze()

    # for observed data, create fake non-arrival timestamps
    if BYR_DELAYS not in data:
        ranges = days[days > 0].apply(range)
        wide = pd.DataFrame.from_records(ranges.values, index=ranges.index)
        s = wide.rename_axis('day', axis=1).stack().astype('int64')
        ts_delay = s * DAY + data[LOOKUP][START_TIME]
        ts_delay += (np.random.uniform(size=len(ts_delay)) * DAY).astype('int64')

    # for agent data, match format and index for recorded delays
    else:
        ts_delay = data[BYR_DELAYS][CLOCK].to_frame().assign(index=1)
        byr_threads = data[X_THREAD].loc[data[X_THREAD][BYR_AGENT], BYR_AGENT]
        byr_threads = byr_threads.reset_index(THREAD)[THREAD]
        ts_delay = ts_delay.join(byr_threads)
        ts_delay.loc[ts_delay[THREAD].isna(), THREAD] = 0
        ts_delay[THREAD] = ts_delay[THREAD].astype('uint64')
        ts_delay = ts_delay.set_index([THREAD, INDEX], append=True).squeeze()
        ts_delay.index = ts_delay.index.reorder_levels(first_offer.index.names)
        ts_delay = ts_delay.sort_index()

    # concatenate with offers
    ts_offer = pd.concat([first_offer, later_offers]).sort_index()
    data[CLOCK] = pd.concat([ts_delay, ts_offer]).sort_index()

    return ts_delay.index, ts_offer.index


def process_byr_inputs(data=None):
    # append arrival timestamps to clock
    idx0, idx1 = append_arrival_delays(data)

    # reshape offers dataframe
    reshape_offers(data=data, idx0=idx0, idx1=idx1)

    # buyer index
    idx = data[CLOCK][data[CLOCK].index.isin(IDX[BYR], level=INDEX)].index

    # outcome
    y = (data[X_OFFER].loc[idx, CON] * CON_MULTIPLIER).astype('int8')

    # input feature dictionary
    x = construct_x(idx=idx, data=data)

    # indices for listing features
    idx_x = get_ind_x(lstgs=data[LOOKUP].index, idx=idx)

    return {'y': y, 'x': x, 'idx_x': idx_x}


def main():
    print('{}/{}'.format(VALIDATION, POLICY_BYR))

    # input dataframes, output processed dataframes
    data = load_data(part=VALIDATION)
    d = process_byr_inputs(data)

    # save various output files
    save_files(d, VALIDATION, POLICY_BYR)


if __name__ == '__main__':
    main()

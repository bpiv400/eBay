import pandas as pd
from inputs.util import save_files, get_x_thread, get_x_offer_init, \
    get_ind_x
from processing.util import extract_day_feats
from utils import input_partition, load_file
from constants import IDX, DAY, MONTH, BYR, CON_MULTIPLIER, POLICY_BYR
from featnames import CON, START_TIME, EXP, DELAY, THREAD_COUNT, \
    MONTHS_SINCE_LSTG, LOOKUP


def construct_x(idx=None, threads=None, offers=None,
                clock=None, lstg_start=None):
    # thread features
    x_thread = get_x_thread(threads, idx,
                            turn_indicators=True)
    x_thread.drop(THREAD_COUNT, axis=1, inplace=True)

    # split master index
    idx1 = pd.Series(index=idx).xs(
        1, level='index', drop_level=False).index
    idx2 = idx.drop(idx1)

    # current time feats
    clock1 = pd.Series(DAY * idx1.get_level_values(level='day'),
                       index=idx1) + lstg_start.reindex(index=idx1,
                                                        level='lstg')
    clock2 = clock.groupby(clock.index.names[:-1]).shift()
    clock2 = clock2[idx2].astype(clock.dtype)
    combined = pd.concat([clock1, clock2]).sort_index()
    date_feats = extract_day_feats(combined).rename(
        lambda c: 'thread_{}'.format(c), axis=1)

    # thread features
    x_thread = pd.concat([date_feats, x_thread], axis=1)

    # redefine months_since_lstg
    x_thread.loc[idx1, MONTHS_SINCE_LSTG] = \
        idx1.get_level_values(level='day') * DAY / MONTH

    # initialize x with thread features
    x = {'thread': x_thread}

    # offer features
    new_idx = idx.to_frame().xs(0, level='day').index
    x_offer = get_x_offer_init(offers, new_idx, role=BYR)
    x_offer = {k: v.reindex(index=idx, fill_value=0.)
               for k, v in x_offer.items()}
    x.update(x_offer)

    # no nans
    for v in x.values():
        assert v.isna().sum().sum() == 0

    return x


def get_y(idx, idx1, offers):
    y = pd.Series(0, dtype='int8', index=idx)
    y.loc[idx1] = (offers[CON] * CON_MULTIPLIER).astype(y.dtype)
    return y


def create_index(clock=None, offers=None, lstg_start=None):
    # buyer turns
    s = clock[clock.index.isin(IDX[BYR], level='index')]
    # remove censored
    censored = offers[EXP] & (offers[DELAY] < 1)
    s = s[~censored]
    # hours before first offer
    arrival_time = s.xs(1, level='index', drop_level=False)
    days = ((arrival_time - lstg_start) // DAY).rename('day')
    days0 = days[days > 0] - 1
    wide = days0.to_frame().assign(con=0).set_index(
        'day', append=True).squeeze().unstack()
    for i in wide.columns:
        wide.loc[i < days0, i] = 0.
    idx0 = wide.stack().index
    # combine with offers index
    idx1 = pd.MultiIndex.from_frame(
        days.reindex(index=s.index, fill_value=0).reset_index())
    idx, _ = idx0.union(idx1).sortlevel()
    return idx, idx1


def process_inputs(part):
    # load dataframes
    offers = load_file(part, 'x_offer')
    threads = load_file(part, 'x_thread')
    clock = load_file(part, 'clock')
    lookup = load_file(part, LOOKUP)
    lstg_start = lookup[START_TIME]

    # master index
    idx, idx1 = create_index(clock=clock,
                             offers=offers,
                             lstg_start=lstg_start)

    # outcome
    y = get_y(idx, idx1, offers)

    # input feature dictionary
    x = construct_x(idx=idx,
                    threads=threads,
                    offers=offers,
                    clock=clock,
                    lstg_start=lstg_start)

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

import pandas as pd
from inputs.util import save_files, get_x_thread, \
    get_x_offer_init, get_init_data
from processing.util import extract_day_feats
from utils import input_partition, init_x
from constants import IDX, DAY, BYR, CON_MULTIPLIER, \
    TRAIN_MODELS, VALIDATION, TEST
from featnames import CON, START_TIME, EXP, DELAY, THREAD_COUNT, \
    MONTHS_SINCE_LSTG


def construct_x(part=None, idx=None, data=None):
    # listing features
    x = init_x(part, idx)
    del x['slr']  # byr does not observe slr features

    # remove auto accept/reject features from x['lstg'] for buyer models
    x['lstg'].drop(['auto_decline', 'auto_accept',
                    'has_decline', 'has_accept',
                    'lstg_ct', 'bo_ct'],
                   axis=1, inplace=True)

    # thread features
    x_thread = get_x_thread(data['threads'], idx,
                            turn_indicators=True)
    x_thread.drop(THREAD_COUNT, axis=1, inplace=True)

    # split master index
    idx1 = pd.Series(index=idx).xs(
        1, level='index', drop_level=False).index
    idx2 = idx.drop(idx1)

    # current time feats
    lstg_start = data['lookup'][START_TIME]
    clock1 = pd.Series(DAY * idx1.get_level_values(level='day'),
                       index=idx1) + lstg_start.reindex(index=idx1,
                                                        level='lstg')
    clock = data['clock']
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

    # concatenate with the listing grouping
    x['lstg'] = pd.concat([x['lstg'], x_thread], axis=1)

    # offer features
    x.update(get_x_offer_init(data['offers'], idx,
                              role=BYR, delay=True))

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
    data = get_init_data(part)

    # master index
    idx, idx1 = create_index(clock=data['clock'],
                             offers=data['offers'],
                             lstg_start=data['lookup'][START_TIME])

    # outcome
    y = get_y(idx, idx1, data['offers'])

    # input feature dictionary
    x = construct_x(part=part, idx=y.index, data=data)

    return {'y': y, 'x': x}


def main():
    # extract parameters from command line
    part = input_partition()
    name = 'policy_{}_delay'.format(BYR)
    print('%s/%s' % (part, name))

    # policy is trained on TRAIN_MODELS
    assert part in [TRAIN_MODELS, VALIDATION, TEST]

    # input dataframes, output processed dataframes
    d = process_inputs(part)

    # save various output files
    save_files(d, part, name)


if __name__ == '__main__':
    main()

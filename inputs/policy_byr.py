import pandas as pd
from inputs.inputs_utils import save_files, get_x_thread, \
    calculate_remaining, get_x_offer_init
from processing.processing_utils import extract_day_feats
from utils import load_file, init_x, input_partition
from constants import IDX, BYR_PREFIX, CON_MULTIPLIER, \
    TRAIN_MODELS, VALIDATION, TEST, DAY, MONTH
from featnames import CON, INT_REMAINING, START_TIME, MONTHS_SINCE_LSTG


def get_y(df, lstg_start):
    # waiting periods before first offer
    arrival_time = df.clock.xs(1, level='index', drop_level=False)
    days = ((arrival_time - lstg_start) // DAY).rename('day')
    # expand index
    days0 = days[days > 0] - 1
    wide = days0.to_frame().assign(con=0).set_index(
        'day', append=True).squeeze().unstack()
    for i in wide.columns:
        wide.loc[i < days0, i] = 0.
    y0 = wide.stack().astype('int8')
    # combine concession and waiting time
    y = (df[CON] * CON_MULTIPLIER).astype('int8')
    y = y.to_frame().join(days.reindex(
        index=y.index, fill_value=0)).set_index(
        'day', append=True).squeeze()
    y = pd.concat([y0, y]).sort_index()
    return y


def process_inputs(part):
    # load dataframes
    offers = load_file(part, 'x_offer')
    threads = load_file(part, 'x_thread')
    clock = load_file(part, 'clock')
    lstg_start = load_file(part, 'lookup')[START_TIME]

    # restict by role, join timestamps and offer features
    df = clock[clock.index.isin(IDX[BYR_PREFIX], level='index')]
    df = df.to_frame().join(offers)

    # outcome and master index
    y = get_y(df, lstg_start)
    idx = y.index
    idx1 = y.xs(1, level='index', drop_level=False).index
    idx2 = idx.drop(idx1)

    # listing features
    x = init_x(part, idx)

    # remove auto accept/reject features from x['lstg'] for buyer models
    x['lstg'].drop(['auto_decline', 'auto_accept',
                    'has_decline', 'has_accept'],
                   axis=1, inplace=True)

    # current time feats
    clock1 = pd.Series(DAY * idx1.get_level_values(level='day'),
                       index=idx1) + lstg_start.reindex(index=idx1,
                                                        level='lstg')
    clock2 = clock.xs(1, level='index').reindex(index=idx2)
    date_feats = pd.concat([extract_day_feats(clock1),
                            extract_day_feats(clock2)]).sort_index()
    date_feats.rename(lambda c: 'thread_{}'.format(c),
                      axis=1, inplace=True)

    # thread features
    x_thread = date_feats.join(get_x_thread(threads, idx,
                                            turn_indicators=True))

    # redefine months_since_lstg
    x_thread.loc[idx1, MONTHS_SINCE_LSTG] = \
        idx1.get_level_values(level='day') * DAY / MONTH

    # remaining
    x_thread[INT_REMAINING] = \
        calculate_remaining(lstg_start=lstg_start,
                            clock=clock,
                            idx=idx)
    x['lstg'] = pd.concat([x['lstg'], x_thread], axis=1)

    # offer features
    x.update(get_x_offer_init(offers, idx, role=BYR_PREFIX, delay=True))

    return {'y': y, 'x': x}


def main():
    # extract parameters from command line
    part = input_partition()
    name = 'policy_{}_delay'.format(BYR_PREFIX)
    print('%s/%s' % (part, name))

    # policy is trained on TRAIN_MODELS
    assert part in [TRAIN_MODELS, VALIDATION, TEST]

    # input dataframes, output processed dataframes
    d = process_inputs(part)

    # save various output files
    save_files(d, part, name)


if __name__ == '__main__':
    main()

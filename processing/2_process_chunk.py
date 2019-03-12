"""
For each chunk of data, create simulator and RL inputs.
"""

import argparse, pickle
import numpy as np, pandas as pd
#from time_feats import get_time_feats

ORIGIN = '2012-06-01 00:00:00'


def reshape_long(df, cols):
    z = df[cols]
    if cols != [1, 2, 3]:
        z = z.rename(columns={cols[0]: 1, cols[1]: 2, cols[2]: 3})
    z = z.stack(dropna=False)
    z.index.rename(['thread','turn'], inplace=True)
    return z


def trim_threads(v):
    keep = v.isna().groupby(level='thread').sum() < 3
    return v[keep[v.index.get_level_values('thread')].values]


def add_offer_vars(x_offer, d, cols, prefix):
    for key, val in d.items():
        x_offer['_'.join([prefix, key])] = reshape_long(val, cols)
    if prefix == 's_curr':
        x_offer.drop('s_curr_msg', axis=1, inplace=True)
    return x_offer


def get_x_offer(byr, slr):
    """
    Creates a df with simulator input at each turn.

    Categories of output variables:
        1. prev slr offer
        2. curr byr offer
        3. curr slr offer
    Index: [thread_id, turn] (turn in range 1...3)
    """
    idx = pd.MultiIndex.from_product(
        [byr['con'].index.values, [1, 2, 3]], names=['thread', 'turn'])
    x_offer = pd.DataFrame(index=idx)
    x_offer = add_offer_vars(x_offer, slr, [1, 2, 3], 's_prev')
    x_offer = add_offer_vars(x_offer, byr, [1, 2, 3], 'b')
    x_offer = add_offer_vars(x_offer, slr, [2, 3, 4], 's_curr')
    return x_offer


def get_y(slr):
    '''
    Creates a dataframe of slr delays, concessions and msg indicators.
    '''
    y = {}
    # delay
    y['delay'] = reshape_long(slr['days'], [2, 3, 4]) / 2
    y['delay'].loc[y['delay'] == 0.] = np.nan     # automatic responses
    # concession
    y['con'] = reshape_long(slr['con'], [2, 3, 4])
    y['con'].loc[np.isnan(y['delay'])] = np.nan  # non-exsitent offers
    y['con'].loc[y['delay'] == 1.] = np.nan  # expired offers
    # message
    y['msg'] = reshape_long(slr['msg'], [2, 3, 4])
    y['msg'].loc[y['con'].isna()] = np.nan    # no concession
    y['msg'].loc[y['con'] == 1.] = np.nan    # accepts
    # trim threads
    return {k: trim_threads(v) for k, v in y.items()}


def split_byr_slr(df):
    b = df[[1, 3, 5, 7]].rename(columns={3:2, 5:3, 7:4})
    s = df[[0, 2, 4, 6]].rename(columns={0:1, 4:3, 6:4})
    return b, s


def get_days(timestamps):
    # initialize dataframes
    b_days = pd.DataFrame(index=timestamps.index, columns=[1, 2, 3, 4])
    s_days = pd.DataFrame(index=timestamps.index, columns=[1, 2, 3, 4])
    # first delay
    s_days[1] = 0.
    # remaining delays
    for i in range(1, 8):
        sec = timestamps[i] - timestamps[i-1]
        sec = sec.dt.total_seconds().astype(np.float64)
        sec[sec <= 1] = 0
        if i % 2:
            b_days[int((i+1)/2)] = sec / (24 * 3600)
        else:
            sec[sec > 48 * 3600] = 48 * 3600
            s_days[int(1 + i/2)] = sec / (24 * 3600)
    return b_days, s_days


def get_con(offers):
    '''
    Creates dataframes of concessions at each turn,
    separately for buyer and seller.
    '''
    # initialize dataframes
    b_con = pd.DataFrame(index=offers.index, columns=[1, 2, 3, 4])
    s_con = pd.DataFrame(index=offers.index, columns=[1, 2, 3, 4])
    # first concession
    b_con[1] = offers[1] / offers[0]
    s_con[1] = 0
    assert np.count_nonzero(np.isnan(b_con[1].values)) == 0
    # remaining concessions
    for i in range(2, 8):
        norm = (offers[i] - offers[i-2]) / (offers[i-1] - offers[i-2])
        if i % 2:
            b_con[int((i+1)/2)] = norm
        else:
            s_con[int(1 + i/2)] = norm
    # verify that all concessions are in bounds
    assert np.nanmax(b_con.values) <= 1 and np.nanmin(b_con.values) >= 0
    assert np.nanmax(s_con.values) <= 1 and np.nanmin(s_con.values) >= 0
    return b_con, s_con


def get_role_vars(timestamps, offers, msgs):
    b_time, s_time = split_byr_slr(timestamps)
    b_days, s_days = get_days(timestamps)
    b_offer, s_offer = split_byr_slr(offers)
    b_con, s_con = get_con(offers)
    b_msg, s_msg = split_byr_slr(msgs)
    byr = {'time': b_time,
           'days': b_days,
           'offer': b_offer,
           'con': b_con,
           'msg': b_msg}
    slr = {'time': s_time,
           'days': s_days,
           'offer': s_offer,
           'con': s_con,
           'msg': s_msg}
    return byr, slr


def split_var(O, T, name):
    df = O[name].unstack()
    if name == 'price':
        df[0] = T.start_price
    elif name == 'message':
        df[0] = 0.
    elif name == 'clock':
        t0 = pd.to_datetime(T.start_time, unit='s', origin=ORIGIN)
        for col in df.columns:
            df[col] = t0 + pd.to_timedelta(df[col], unit='s')
        df[0] = pd.to_datetime(T.start_date, unit='D', origin=ORIGIN)
    return df.sort_index(axis=1)


if __name__ == "__main__":
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num

    # load data
    path = 'data/chunks/%d.pkl' % num
    print("Loading data")
    chunk = pickle.load(open(path, 'rb'))
    L, T, O = [chunk[k] for k in ['listings', 'threads', 'offers']]
    L.rename(columns={'start':'start_date'}, inplace=True)
    T.rename(columns={'start':'start_time'}, inplace=True)
    T = T.join(L, on='lstg')

    # ids of threads to keep
    threads = T[(T.bin_rev == 0) & (T.flag == 0)].index.sort_values()

    # time-varying features
    print("Calculating time features")
    #time_feats = get_time_feats(O, T, L)

    # attributes
    print("Calculating attributes")
    timestamps = split_var(O, T, 'clock').loc[threads]
    offers = split_var(O, T, 'price').loc[threads]
    msgs = split_var(O, T, 'message').loc[threads]

    # byr and slr dictionaries
    byr, slr = get_role_vars(timestamps, offers, msgs)

    # simulator outcomes
    print("Calculating outcomes")
    y = get_y(slr)

    # offer features
    print("Calculating offer features")
    x_offer = get_x_offer(byr, slr)

    # write simulator chunk
    print("Writing first chunk")
    chunk = {'y': y,
             'x_offer': x_offer,
             'T': T.loc[threads]}
    path = 'data/chunks/%d_simulator.pkl' % num
    pickle.dump(chunk, open(path, 'wb'))

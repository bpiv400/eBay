from datetime import datetime as dt
import pandas as pd
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial


def date_feats(feat_df):
    # grab offer time
    off_series = feat_df['src_cre_date'].values
    # grab auction post time
    post_series = feat_df['auct_start_dt'].values
    # grab auction expiration time
    close_series = feat_df['auct_end_dt'].values

    # convert all to dates
    post_series = pd.to_datetime(post_series).values.astype('datetime64[D]')
    close_series = pd.to_datetime(close_series).values.astype('datetime64[D]')
    off_series = pd.to_datetime(off_series).values.astype('datetime64[D]')

    # find the duration of the auction (inclusive)
    # (expiration time - post time)
    dur = np.busday_count(post_series, close_series,
                          weekmask=[1, 1, 1, 1, 1, 1, 1]) + 1

    # find time until the auction closes, remaining time,\
    #  (expiration time - offer time)
    rem = np.busday_count(off_series, close_series,
                          weekmask=[1, 1, 1, 1, 1, 1, 1])

    # convert time until auction close to fraction of
    # total duration (remaining time / duration)
    rem = np.divide(rem, dur)

    feat_df['dur_time'] = dur
    feat_df['rem_time'] = rem
    return feat_df


def norm_price(feat_df):
    feat_df['norm_offr_price'] = np.divide(
        feat_df['offr_price'], feat_df['start_price_usd'])
    feat_df['norm_rsp_price'] = np.divide(
        feat_df['rsp_offr'], feat_df['start_price_usd'])
    return feat_df


def main():
    parser = argparse.ArgumentParser(
        description='associate threads with all relevant variables')
    parser.add_argument('--name', action='store', type=str)
    args = parser.parse_args()
    filename = args.name
    print('Reading csv')
    feat_df = pd.read_csv('data/' + filename + '_feats.csv')
    print('Creating price normalization features')
    feat_df = norm_price(feat_df)
    print('Creating Date feature')
    feat_df = date_feats(feat_df)
    print('Writing csv')
    feat_df.to_csv('data/' + filename + '_feats.csv')


if __name__ == '__main__':
    main()

from datetime import datetime as dt
import pandas as pd
import numpy as np


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

    # find the duration of the auction

    # (expiration time - post time)

    # find time until the auction closes, remaining time,\
    #  (expiration time - offer time)

    # convert time until auction close to fraction of
    # total duration (remaining time / duration)

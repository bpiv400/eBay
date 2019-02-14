"""
Imputes values associated with seller rejections
"""
import argparse
import numpy as np
import pandas as pd
from util_env import extract_datatype


def impute_rejections(df):
    '''
    Adds rows for seller rejections in each thread

    Args:
        df: dataframe containing standard threads file
    Returns:
        updated dataframe
    '''
    df


def main():
    """
    Main method
    """
    parser = argparse.ArgumentParser()
    # name gives the chunk name (e.g. train-1)
    parser.add_argument('--name', action='store', required=True)
    chunk_name = parser.parse_args().name
    datatype = extract_datatype(chunk_name)
    offrs_path = 'data/%s/threads/%s_threads.csv' % (dataype, chunk_name)
    df = pd.read_pickle(offrs_path)

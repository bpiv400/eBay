import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn as sk
import sys
import os
import argparse


def load_data():
    df = pd.read_csv('data/curr_exp/' + name + '_concat.csv',
    parse_dates=['src_cre_date', 'auct_start_dt',
                                       'auct_end_dt', 'response_time'],
                          dtype={'unique_thread_id': np.int64})


def main():
    # parse parameters
     parser = argparse.ArgumentParser(
        description='associate threads with all relevant variables')
    parser.add_argument('--name', action='store', type=str)
    args=parser.parse_args()
    filename=args.name
    df=load_data()
    # initialize model
    model=sk.linear_model.LogisticRegression(penalty='L2', solver='lbfgs', max_iter=100,
                                               verbose=1, C=10)


if __name__ == '__main__':
    main()

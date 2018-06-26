import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.linear_model as lm
import sys
import os
import argparse


def load_data(name):
    df = pd.read_csv('data/curr_exp/' + name + '_concat.csv',
                     dtype={'unique_thread_id': np.int64})
    return df

# checks if the column has any nan values


def has_nan(df, colname):
    vals = df[colname].values
    has_nan = np.isnan(vals).any()
    print('%s nans: %r' % (colname, has_nan))


def main():
    # parse parameters
    parser = argparse.ArgumentParser(
        description='associate threads with all relevant variables')
    parser.add_argument('--name', action='store', type=str)
    args = parser.parse_args()
    filename = args.name
    df = load_data(filename)

    # drop any

    # initialize model
    model = lm.LogisticRegression(penalty='L2', solver='lbfgs', max_iter=100,
                                  verbose=1, C=10)
    # extract response variable
    targ_vect = df['resp_offr'].values
    df.drop(columns=['resp_offr'], inplace=True)
    print('Hiding nan\'s: %r' % df.isnull().values.any())
    cols = df.columns.values
    col_dict = {}
    cntr = 0
    for col in cols:
        col_dict[col] = cntr
        has_nan(df, col)
        cntr = cntr + 1
    df = df.values

    # model.fit(df, y)
if __name__ == '__main__':
    main()

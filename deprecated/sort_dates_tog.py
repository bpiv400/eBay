import random
import numpy as np
import pandas as pd
import sys


def sort(df):
    df.sort_values(by='src_cre_date', ascending=True,
                   inplace=True)
    return df


def main():
    files = ['toy.csv', 'train.csv', 'pure_test.csv', 'test.csv']
    for f in files:
        print('Loading %s' % f)
        sys.stdout.flush()
        data = pd.read_csv('data/' + f)
        print('Converting %s to datetime' % f)
        sys.stdout.flush()
        data['src_cre_date'] = pd.to_datetime(data.src_cre_date)
        print('Ordering %s by date together' % f)
        data = sort(data)
        print('Saving %s' % f)
        sys.stdout.flush()
        data.to_csv('data/sorted_tog/' + f)
    print('Done')


if __name__ == '__main__':
    main()

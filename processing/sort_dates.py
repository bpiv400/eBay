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
        print('Ordering %s by date in groups' % f)
        sys.stdout.flush()
        data = data.groupby(by='unique_thread_id')
        data = data.apply(sort)
        print('Saving %s' % f)
        sys.stdout.flush()
        data.to_csv('data/sorted/' + f)
    print('Done')


if __name__ == '__main__':
    main()

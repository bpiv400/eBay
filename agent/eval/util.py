import argparse
import os
import pandas as pd
from utils import topickle, unpickle
from agent.const import DELTA_CHOICES
from constants import NUM_CHUNKS
from featnames import AGENT_PARTITIONS, TEST, INDEX


def sim_args(num=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str,
                        choices=AGENT_PARTITIONS, default=TEST)
    parser.add_argument('--heuristic', action='store_true')
    parser.add_argument('--delta', type=float, choices=DELTA_CHOICES)
    if num:
        parser.add_argument('--num', type=int,
                            choices=range(1, NUM_CHUNKS + 1))
    return parser.parse_args()


def print_table(df):
    cols = list(df.columns)
    print(df[[c for c in cols if 'buy' not in c]])
    newdf = df[[c for c in cols if 'buy' in c]].rename(
        lambda x: x.replace('_{}'.format('buy'), ''), axis=1)
    if len(newdf.columns) > 0:
        print(newdf)


def read_table(run_dir=None):
    path = run_dir + '{}.pkl'.format(TEST)
    if os.path.isfile(path):
        df = unpickle(path)
        print_table(df)
        exit()


def save_table(run_dir=None, output=None):
    df = pd.DataFrame.from_dict(output, orient=INDEX)
    topickle(df, run_dir + '{}.pkl'.format(TEST))
    print_table(df)

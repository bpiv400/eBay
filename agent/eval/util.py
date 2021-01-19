import argparse
import os
import pandas as pd
from agent.util import load_valid_data, get_output_dir
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
    print(df[[c for c in cols if '_' not in c]])
    for name in ['offer', 'sale']:
        newdf = df[[c for c in cols if name in c]].rename(
            lambda x: x.replace('_{}'.format(name), ''), axis=1)
        if len(newdf.columns) > 0:
            print(newdf)


def read_table(run_dir=None):
    path = run_dir + '{}.pkl'.format(TEST)
    if os.path.isfile(path):
        df = unpickle(path)
        print_table(df)
        exit()


def collect_output(run_dir=None, delta=None, f=None):
    byr = delta is None
    output = dict()

    # rewards from data
    data = load_valid_data(part=TEST, byr=byr)
    output['Humans'] = f(data)

    # rewards from heuristic strategy
    heuristic_dir = get_output_dir(heuristic=True, part=TEST, delta=delta)
    data = load_valid_data(part=TEST, run_dir=heuristic_dir)
    if data is not None:
        output['Heuristic'] = f(data)

    # rewards from agent run
    data = load_valid_data(part=TEST, run_dir=run_dir)
    if data is not None:
        output['Agent'] = f(data)

    # convert to table and save
    df = pd.DataFrame.from_dict(output, orient=INDEX)
    topickle(df, run_dir + '{}.pkl'.format(TEST))
    print_table(df)

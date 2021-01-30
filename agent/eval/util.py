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
    parser.add_argument('--byr', action='store_true')
    parser.add_argument('--delta', type=float, choices=DELTA_CHOICES)
    if num:
        parser.add_argument('--num', type=int,
                            choices=range(1, NUM_CHUNKS + 1))
    return parser.parse_args()


def read_table(run_dir=None):
    path = run_dir + '{}.pkl'.format(TEST)
    if os.path.isfile(path):
        df = unpickle(path)
        print(df)
        exit()


def save_table(run_dir=None, output=None):
    df = pd.DataFrame.from_dict(output, orient=INDEX)
    topickle(df, run_dir + '{}.pkl'.format(TEST))
    read_table(run_dir=run_dir)

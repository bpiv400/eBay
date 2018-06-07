import pandas as pd
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count

parser = argparse.ArgumentParser(
    description='associate threads with all relevant variables')
parser.add_argument('--name', action='store', type=str)
args = parser.parse_args()
filename = args.name

data = pd.read_csv('data/' + filename)
lists = pd.read_csv('data/lists.csv')


def parApply(groupedDf, func):
    with Pool(cpu_count()) as p:
        ret_list = p.map(func, [group for name, group in groupedDf])
    return pd.concat(ret_list)

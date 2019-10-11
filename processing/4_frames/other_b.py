import sys
import argparse, random
from compress_pickle import load, dump
from datetime import datetime as dt
from sklearn.utils.extmath import cartesian
import numpy as np, pandas as pd

sys.path.append('repo/')
from constants import *
from utils import *

sys.path.append('repo/processing/')
from processing_utils import *


if __name__ == "__main__":
    # partition number from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num

    # partition
    partitions = load(PARTS_DIR + 'partitions.gz')
    part = list(partitions.keys())[num-1]
    idx = partitions[part]
    path = lambda name: '%s/%s/%s.gz' % (PARTS_DIR, part, name)

    x_offer = load(path('x_offer')).sort_index()
    dump(x_offer, path('x_offer'))
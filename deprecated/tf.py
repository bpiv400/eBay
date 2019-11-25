import sys
import argparse, random
from compress_pickle import load, dump
from datetime import datetime as dt
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing.processing_utils import *


if __name__ == "__main__":
    # partition number from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num-1

    # partition
    part = PARTITIONS[num]
    idx, path = get_partition(part)

    # differenced time features
    print('tf_role_diff')
    tf_role_diff = load_frames('tf_con').reindex(
        index=idx, level='lstg')
    dump(tf_role_diff, path('tf_role_diff'))

    # raw time features
    print('tf_role_raw')
    tf_role_raw = load_frames('tf_delay_raw').reindex(
        index=idx, level='lstg')
    dump(tf_role_raw, path('tf_role_raw'))
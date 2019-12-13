import sys, argparse
from compress_pickle import load, dump
import numpy as np, pandas as pd
from constants import *
from utils import *
from processing.processing_utils import *


if __name__ == "__main__":
    # parameter(s) from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', action='store', type=str, required=True)
    part = parser.parse_args().part
    idx, path = get_partition(part)

    # slr features
    print('Seller features')
    slr = load_frames('slr').reindex(index=idx, fill_value=0)
    dump(slr, path('x_slr'))
    del slr

    # cat and cndtn features
    print('Categorical features')
    df = load_frames('cat').reindex(index=idx, fill_value=0)
    for name in ['cat', 'cndtn']:
    	x = df[[c for c in df.columns if c.startswith(name + '_')]]
    	dump(x, path('x_' + name))
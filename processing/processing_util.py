import pickle, numpy as np, pandas as pd
from constants import *


def load_frames(name):
    # list of paths
    paths = ['%s/%s' % (CHUNKS_DIR, p) for p in os.listdir(CHUNKS_DIR)
        if os.path.isfile('%s/%s' % (CHUNKS_DIR, p)) and name in p]
    # loop and append
    df = pd.DataFrame()
    for path in sorted(paths):
        stub = pickle.load(open(path, 'rb'))
        df = df.append(stub)
        del stub
    return df


def multiply_indices(s):
    # initialize arrays
    k = len(s.index.names)
    arrays = np.zeros((s.sum(),k+1), dtype=np.int64)
    count = 0
    # outer loop: range length
    for i in range(1, max(s)+1):
        index = s.index[s == i].values
        if len(index) == 0:
            continue
        # cartesian product of existing level(s) and period
        if k == 1:
            f = lambda x: cartesian([[x], list(range(i))])
        else:
            f = lambda x: cartesian([[e] for e in x] + [list(range(i))])
        # inner loop: rows of period
        for j in range(len(index)):
            arrays[count:count+i] = f(index[j])
            count += i
    # convert to multi-index
    return pd.MultiIndex.from_arrays(np.transpose(arrays), 
        names=s.index.names + ['period'])
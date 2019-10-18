from compress_pickle import load, dump
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np, pandas as pd

from constants import *
from utils import *
from processing.processing_utils import *


# partitions dataframe on lstg using partitions dictionary
def partition_frame(partitions, df, name):
    for part, idx in partitions.items():
        if len(df.index.names) == 1:
            toSave = df.reindex(index=idx)
        else:
            toSave = df.reindex(index=idx, level='lstg')
        dump(toSave, PARTS_DIR + part + '/' + name + '.gz')


# scales variables, does PCA, keeps components with 95% of variance
def do_pca(df):
    # standardize variables
    vals = StandardScaler().fit_transform(df)
    # PCA
    N = len(df.columns)
    pca = PCA(n_components=N, svd_solver='full')
    components = pca.fit_transform(vals)
    # select number of components
    shares = np.var(components, axis=0) / N
    keep = 1
    while np.sum(shares[:keep]) < PCA_CUTOFF:
        keep += 1
    # return dataframe
    return pd.DataFrame(components[:,:keep], index=df.index, 
        columns=['c' + str(i) for i in range(1,keep+1)])


if __name__ == "__main__":
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', action='store', type=int, required=True)
    num = parser.parse_args().num

    # load partitions
    partitions = load(PARTS_DIR + 'partitions.gz')

    # do pca on time-valued features
    if num == 1:
        # slr
        print('slr time-valued features')
        tf_slr = load_frames('tf_slr')
        tf_slr = do_pca(tf_slr)
        partition_frame(partitions, tf_slr, 'x_slr')
    elif num == 2:
        # meta
        print('meta time-valued features')
        idx = np.sort(np.concatenate(list(partitions.values())))
        tf_meta = []
        for i in range(N_META):
            tf_meta.append(load(FEATS_DIR + 'm' + str(i) + '_tf_meta.gz'))
        tf_meta = pd.concat(tf_meta).reindex(index=idx)
        tf_meta = do_pca(tf_meta)
        partition_frame(partitions, tf_meta, 'x_meta')

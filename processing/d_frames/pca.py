import sys, argparse
from compress_pickle import load, dump
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from constants import *
from utils import *
from processing.processing_utils import *


# scales variables and performs PCA
def do_pca(df, pre):
    # standardize variables
    vals = StandardScaler().fit_transform(df)
    # PCA
    N = len(df.columns)
    pca = PCA(n_components=N, svd_solver='full')
    components = pca.fit_transform(vals)
    # return dataframe
    return pd.DataFrame(components, index=df.index, 
        columns=['%s%d' % (pre, i) for i in range(N)])


if __name__ == "__main__":
	# 1 for w2v, 2 for slr, 3 for cat
	parser = argparse.ArgumentParser()
	parser.add_argument('--num', action='store', type=int, required=True)
	num = parser.parse_args().num

	# load partitions and concatenate indices
	partitions = load(PARTS_DIR + 'partitions.gz')
	idx = np.sort(np.concatenate(list(partitions.values())))

	# load data
	if num == 1:
		name = 'w2v'
		s = load(CLEAN_DIR + 'listings.pkl')[['cat']].reindex(index=idx)
		byr = load(W2V_DIR + 'byr.gz').reindex(
			index=s.values.squeeze(), fill_value=0)
		slr = load(W2V_DIR + 'slr.gz').reindex(
			index=s.values.squeeze(), fill_value=0)
		var = pd.concat([byr, slr], axis=1)
		var.set_index(s.index, inplace=True)
	else:
		name = 'slr' if num == 2 else 'cat'
		var = load_frames(name).reindex(index=idx, fill_value=0)

	# pca
	var = do_pca(var, name)

	# save by partition
	for part, indices in partitions.items():
		dump(var.reindex(index=indices), 
			PARTS_DIR + '%s/x_%s.gz' % (part, name))
